# lives data exists in the following format:
#   trs files and audio recordings
#   recordings are long and include multiple participants
#   this should be prepared as a structured dialog task
#   so that you can examine each item individually, but can also
#   make use of the structure of entire conversations
#   NOTE: the speaker diarization in google is pretty rough
#   todo: this will need to be supported in some way later on
#   todo: what to do about long chunks marked as a single speaker?
#       can split on pauses or can split on num words

from prep_data import *
from utils.data_prep_helpers import (
    get_class_weights,
    get_gender_avgs,
    create_data_folds_list,
    Glove,
    make_glove_dict,
    get_data_samples,
)

from utils.audio_extraction import (
    extract_portions_of_mp4_or_wav,
    convert_to_wav,
    run_feature_extraction,
)


def prep_lives_data(
    data_path="../../datasets/multimodal_datasets/Columbia_deception_corpus",
    feature_set="IS13",
    transcription_type="gold",
    embedding_type="distilbert",
    glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
    features_to_use=None,
    as_dict=False,
    avg_acoustic_data=False,
    custom_feats_file=None,
    num_train_ex=None,
):
    # load glove
    if embedding_type.lower() == "glove":
        glove_dict = make_glove_dict(glove_filepath)
        glove = Glove(glove_dict)
    else:
        glove = None

    # holder for name of file containing utterance info
    utts_name = f"cdc_{transcription_type.lower()}.tsv"

    # create instance of StandardPrep class
    cdc_prep = SelfSplitPrep(
        data_type="cdc",
        data_path=data_path,
        feature_set=feature_set,
        utterance_fname=utts_name,
        glove=glove,
        use_cols=features_to_use,
        as_dict=as_dict,
        avg_acoustic_data=avg_acoustic_data,
        custom_feats_file=custom_feats_file,
        bert_type=embedding_type,
    )

    # get train, dev, test data
    train_data, dev_data, test_data = cdc_prep.get_data_folds()

    # get train ys
    if as_dict:
        train_ys = [int(item["ys"][0]) for item in train_data]
    else:
        train_ys = [int(item[4]) for item in train_data]

    # get updated class weights using train ys
    class_weights = cdc_prep.get_updated_class_weights(train_ys)

    # # get a subset of the training data, if necessary
    if num_train_ex:
        train_data = get_data_samples(train_data, num_train_ex)

    return train_data, dev_data, test_data, class_weights


def preprocess_lives(corpus_path, save_filename, delete_existing_data_file=True):
    """
    Preprocess the lives data files
    This requires extracting times from .trs files,
        getting relevant portions of wav files,
        and feature extraction with openSMILE
    :param corpus_path: the path to the corpus
    :return:
    """
    # if you want to delete existing data, do so here
    if delete_existing_data_file:
        os.remove(f"{corpus_path}/{save_filename}")

    # counter for number of input items
    num_items = 0

    # access path
    for f in glob.glob(f"{corpus_path}/data/*"):
        if f.endswith(".trs"):


        # all R files are participants
        # get transcript files
        if f.endswith("_R_16k.punc.trs"):
            # get the name of the speaker from file name
            speaker = re.search(r"S-(\S+)_R", f.split("/")[-1]).group(1)

            # find the associated ltf file
            ltf_file = "/".join(f.split("/")[:-1])
            ltf_file = f"{ltf_file}/S-{speaker}.ltf"

            # prep the data from these two files
            num_items = prep_cdc_trs_data(
                f, ltf_file, f"{corpus_path}/{save_filename}", speaker, num_items
            )


def prep_cdc_trs_data(
    trs_file_path, ltf_file_path, save_filename_and_path, speaker=None, utt_num=0
):
    """
    Convert CDC trs files to organized tsv
    :return: the new utterance number
    """
    # holder for utterance and gold data
    all_utts = []

    # add header if the file doesn't exist
    if not os.path.exists(save_filename_and_path):
        all_utts.append(
            "speaker\tutt_num\tutterance\ttime_start\ttime_end\ttruth_value"
        )

    # prepare the ltf (gold labeled) data
    ltf_data = []
    with open(ltf_file_path, "r") as ltf_file:
        # skip first two lines
        ltf_file.readlines(2)
        for line in ltf_file:
            # tab separated
            line = line.strip().split("\t")
            # gold label
            truth_val = line[0]
            # start time -- mm:ss.s string
            line_start_time = split_string_time(line[1])
            # end time -- mm:ss.s string
            line_end_time = split_string_time(line[2])
            # add to line
            ltf_data.append([truth_val, line_start_time, line_end_time])

    # open trs file
    with open(trs_file_path, "r") as read_in:
        # pointer to position within ltf_data to reduce redundant searching
        ltf_pointer = 0

        if speaker is None:
            # get the speaker for this file
            speaker = re.search(r"S-(\S+)_R", trs_file_path.split("/")[-1]).group(1)

        # read in all lines but first 6 and last 4
        used_lines = read_in.readlines()[6:-4]

        for i, line in enumerate(used_lines):
            # find lines containing times; this is the start time

            if line.startswith("<Sync time="):
                # clean up the following line
                utt = cdc_string_cleanup(used_lines[i + 1])

                # check if empty
                if len(utt) > 2:
                    # get start time
                    start_time = float(
                        re.search(r'<Sync time="(\S+)"/>', line).group(1)
                    )

                    # get the utt number
                    utt_num += 1

                    try:
                        # get end time -- should usually be two lines after
                        end_time = float(
                            re.search(r'<Sync time="(\S+)"/>', used_lines[i + 2]).group(
                                1
                            )
                        )
                    # if end time isn't where we expect it
                    except AttributeError:
                        end_found = False
                        j = 1
                        # search for end time
                        while not end_found:
                            if used_lines[i + 2 + j].startswith("<Sync time="):
                                end_time = float(
                                    re.search(
                                        r'<Sync time="(\S+)"/>', used_lines[i + 2 + j]
                                    ).group(1)
                                )
                                end_found = True
                            else:
                                j += 1

                    # match start and end times with truth values
                    truth_value = None
                    find_truth = True
                    # find overlap between gold label times and speech times
                    while find_truth:
                        # see if gold label time fully contains speech times
                        if (
                            ltf_data[ltf_pointer][1] <= start_time
                            and ltf_data[ltf_pointer][2] >= end_time
                        ):
                            truth_value = ltf_data[ltf_pointer][0]
                            find_truth = False
                        # see if a new value is added during the speech -- this should be the gold
                        elif start_time <= ltf_data[ltf_pointer][1] <= end_time:
                            truth_value = ltf_data[ltf_pointer][0]
                            find_truth = False
                        # if it's after the last press, it should have the last label
                        elif ltf_pointer == len(ltf_data) - 1:
                            truth_value = ltf_data[ltf_pointer][0]
                            find_truth = False
                        else:
                            ltf_pointer += 1

                    all_utts.append(
                        f"{speaker}\t{utt_num}\t{utt}\t{start_time}\t{end_time}\t{truth_value}"
                    )

    # save the data
    with open(save_filename_and_path, "a") as savefile:
        savefile.write("\n".join(all_utts))
        savefile.write("\n")

    return utt_num
