# prepare the columbia deception corpus
import os
import glob
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from prep_data import SelfSplitPrep, get_updated_class_weights
from utils.data_prep_helpers import (
    split_string_time,
    make_glove_dict,
    Glove,
    get_data_samples,
)
from utils.audio_extraction import (
    extract_portions_of_mp4_or_wav,
    convert_to_wav,
    run_feature_extraction,
)


def prep_cdc_data(
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
    include_spectrograms=False
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
        include_spectrograms=include_spectrograms
    )

    # get train, dev, test data
    train_data, dev_data, test_data = cdc_prep.get_data_folds()

    # get train ys
    if as_dict:
        train_ys = [int(item["ys"][0]) for item in train_data]
    else:
        train_ys = [int(item[4]) for item in train_data]

    # get updated class weights using train ys
    class_weights = get_updated_class_weights(train_ys)

    # # get a subset of the training data, if necessary
    if num_train_ex:
        train_data = get_data_samples(train_data, num_train_ex)

    return train_data, dev_data, test_data, class_weights


def preprocess_cdc(corpus_path, save_filename, delete_existing_data_file=True):
    """
    Preprocess the cdc files
    This requires extracting times from the .ltf files and .trs files,
        getting relevant portions of wav files, and feature extraction
        with openSMILE
    :param corpus_path: the path to the corpus
    :return:
    """
    # if you want to delete existing data, do so here
    if delete_existing_data_file:
        os.remove(f"{corpus_path}/{save_filename}")

    # counter for number of input items
    num_items = 0

    # access path
    for f in glob.glob(f"{corpus_path}/data/*/*"):
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


def cdc_string_cleanup(string):
    """
    Clean up strings with broken tag identifiers from cdc trs
    :param string: the string from trs file
    :return: cleaned up string
    """
    unusable = [
        "&lt;BN&gt;",
        "&lt;BR&gt;",
        "&lt;OTP&gt;",
        "&lt;/OTP&gt;",
        "&lt;SN&gt;",
        "&lt;~&gt;",
        "&lt;LG&gt;",
        "&lt;/OTE&gt;",
        "&lt;OTE&gt;",
        "&lt;MP&gt;",
        "&lt;LN&gt;",
        "/",
    ]
    unusable = "|".join(unusable)
    string = string.strip()
    string = re.sub(unusable, "", string)
    string = re.sub("&lt;UNIN&gt;", "<UNK>", string)
    string = re.sub(r"%[A-Z]+%+\S+%", "", string)

    return string


def extract_audio_for_cdc(gold_label_df, base_path):
    """
    Extract the audio for specified speech spans in gold df
    :param gold_label_df: pandas df containing gold labels, utterances, and times
    :param base_path: path to cdc
    :return:
    """
    # convert all flac files to wav
    all_speakers = set(gold_label_df["speaker"].tolist())
    [
        convert_to_wav(f"{base_path}/data/S-{speaker}/S-{speaker}_R_16k.flac")
        for speaker in all_speakers
    ]

    # extract portions of wav
    for index, row in gold_label_df.iterrows():
        speaker = row["speaker"]
        utt_num = row["utt_num"]
        time_start = row["time_start"]
        time_end = row["time_end"]

        wav_path = f"{base_path}/data/S-{speaker}"
        wav_name = f"S-{speaker}_R_16k.wav"
        save_path = f"{base_path}/wav"
        short_name = f"{speaker}_{utt_num}.wav"

        extract_portions_of_mp4_or_wav(
            wav_path, wav_name, time_start, time_end, save_path, short_name
        )


# if __name__ == "__main__":
#     pass

# corpus_path = "../../datasets/multimodal_datasets/Columbia_deception_corpus"
# save_filename = "cdc_gold.tsv"
# #
# preprocess_cdc(corpus_path, save_filename, delete_existing_data_file=True)
# gold_data_df = pd.read_csv(f"{corpus_path}/{save_filename}", sep="\t")
# extract_audio_for_cdc(gold_data_df, corpus_path)
#
# # run feature extraction
# run_feature_extraction(f"{corpus_path}/wav", "IS13", f"{corpus_path}/IS13")
#
# train, dev, test, weights = prep_cdc_data()
#
# print(train[0])
# print(len(train))
# print(weights)
