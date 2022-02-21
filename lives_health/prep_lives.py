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
import os.path

import pandas as pd

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


def preprocess_lives(corpus_path, flatten_data=False, json_data=True):
    """
    Preprocess the lives data files
    This requires extracting times from .trs files,
        getting relevant portions of wav files,
        and feature extraction with openSMILE
    :param corpus_path: the path to the corpus
    :param flatten_data: whether to utt align data or
        have wd alignment
    :param json_data: whether data is coming from json
        or trs files
    :return:
    """
    # counter for number of input items
    num_items = 0

    # access path
    for f in glob.glob(f"{corpus_path}/*"):
        if f.endswith(".trs") or f.endswith(".json"):
            # split into name of file and name of path
            fpath, fname = f.rsplit("/", 1)

            # make csv converter item from this
            csv_converter = TranscriptToCSV(fpath, fname)

            if json_data:
                if flatten_data:
                    utt_data = csv_converter.convert_json(alignment="utt")
                    save_path = f"{fpath}/transcription/lives_json_gold.csv"
                else:
                    utt_data = csv_converter.convert_json(alignment="word")
                    save_path = f"{fpath}/transcript/lives_json_gold_wordaligned.csv"
            else:
                if flatten_data:
                    trs_df = csv_converter.convert_trs()
                    utt_data = csv_converter.flatten_data(trs_df)

                    # save this utterance-level data
                    save_path = f"{fpath}/transcription/lives_gold.csv"
                else:
                    utt_data = csv_converter.convert_trs()
                    save_path = f"{fpath}/transcript/lives_gold_wordaligned.csv"

            csv_converter.save_data(utt_data, savepath=save_path)


class TranscriptToCSV:
    """
    Takes a trs or json file and converts it to a csv
    Used to preprocess data into the format expected by this repo
    note: trs contains diarization, json does not
    lines of csvfile are speaker,timestart,timeend, word, utt_num where:
        speaker   = caller or patient
        timestart = time of start of turn
        timeend   = time of end of turn
        word      = the word predicted
        utt_num   = the utterance number within the conversation
        word_num  = the word number (useful for averaging over words)
    """

    def __init__(self, path, trsfile, recording2sid=None):
        self.path = path
        self.tname = trsfile
        self.tfile = f"{path}/{trsfile}"

        # a dict of recording_id : sid pairs
        self.recording2sid = recording2sid

    def convert_json(self, alignment="word"):
        """
        Convert json to a format usable in csv files
        :return: the data as a pd dataframe
            this may either be word-aligned or utt-aligned
        """
        with open(self.tfile, "r") as trs:
            tfile_name = self.tfile.split("/")[-1].split(".json")[0]
            if len(self.tfile.split(" ")) > 1:
                participant = tfile_name.split(" ")[0]
                recording_id = tfile_name.split(" ")[1].split(".trs")[0]
            else:
                participant = "TODO"
                recording_id = tfile_name.split(".trs")[0]
                if self.recording2sid is not None:
                    participant = self.recording2sid(recording_id)

            # get transcription json
            whole_transcription = json.load(trs)

            # if alignment.lower() == "word" or alignment.lower() == "wd":
            all_trans = []

            # set new index
            new_idx = 0

            # subtract one because the last item in the transcription is all of the words in the entire thing
            for i in range(len(whole_transcription)):
                if len(whole_transcription[i]['alternatives'][0].keys()) > 1:
                    transcription = pd.DataFrame(whole_transcription[i]['alternatives'][0]['words'])
                    transcription["startTime"] = transcription["startTime"].str.replace(r's', '')
                    transcription["endTime"] = transcription["endTime"].str.replace(r's', '')
                    i += 1
                    transcription["utt_num"] = i
                    transcription = transcription.rename_axis('word_num').reset_index()
                    transcription["word_num"] += 1
                    transcription["speaker"] = "unk"
                    transcription["sid"] = participant
                    transcription["recording_id"] = recording_id

                    # holder for utt info
                    dfs = []

                    # if utterance-aligned
                    if alignment.lower() not in ["word", "wd"]:

                        # calculate different between last work and current work
                        transcription['prev_endTime'] = transcription['endTime'].shift(1)
                        transcription['diff'] = transcription['startTime'].astype(float) - transcription['prev_endTime'].astype(float)

                        # get indices of all rows where diff > .5 seconds
                        all_cutoff_points = transcription[transcription['diff'].gt(.5)].index

                        # if there are any indices
                        if len(all_cutoff_points > 0):
                            last_check = 0
                            # split the dataframe at this point
                            for idx in all_cutoff_points:
                                if all_cutoff_points[-1] != idx:
                                    # get a df of just these rows
                                    this_utt = transcription.loc[last_check:idx - 1]
                                    # get the utterance
                                    sid = this_utt['sid'].iloc[0]
                                    recording_id = this_utt['recording_id'].iloc[0]
                                    # utt_num = this_utt['utt_num'].iloc[0] + new_idx
                                    utt_num = new_idx
                                    speaker = this_utt['speaker'].iloc[0]
                                    start_time= this_utt['startTime'].iloc[0]
                                    end_time = this_utt['endTime'].iloc[-1]
                                    utterance = this_utt['word'].str.cat(sep=" ")
                                    # update utterance number
                                    new_idx += 1
                                    # move forward in index list
                                    last_check = idx
                                else:
                                    this_utt = transcription.loc[idx:]
                                    # get the utterance
                                    sid = this_utt['sid'].iloc[0]
                                    recording_id = this_utt['recording_id'].iloc[0]
                                    # utt_num = this_utt['utt_num'].iloc[0] + new_idx
                                    utt_num = new_idx
                                    speaker = this_utt['speaker'].iloc[0]
                                    start_time = this_utt['startTime'].iloc[0]
                                    end_time = this_utt['endTime'].iloc[-1]
                                    utterance = this_utt['word'].str.cat(sep=" ")
                                    # update utterance number
                                    new_idx += 1
                                # get df
                                dfs.append([sid, recording_id, utt_num, speaker, start_time, end_time, utterance])
                                # print("cutoff points detected--appending:")
                                # print(f"[{sid}, {recording_id}, {utt_num}, ...]")
                        else:
                            # get the utterance
                            sid = transcription['sid'].iloc[0]
                            recording_id = transcription['recording_id'].iloc[0]
                            # utt_num = transcription['utt_num'].iloc[0] + new_idx
                            utt_num = new_idx
                            speaker = transcription['speaker'].iloc[0]
                            start_time = transcription['startTime'].iloc[0]
                            end_time = transcription['endTime'].iloc[-1]
                            utterance = transcription['word'].str.cat(sep=" ")

                            # update utterance number
                            new_idx += 1

                            # get df
                            dfs.append([sid, recording_id, utt_num, speaker, start_time, end_time, utterance])

                        # add list of dataframes to all transcriptions
                        all_trans.extend(dfs)
                    else:
                        # add the transcription to all transcriptions
                        all_trans.append(transcription)

        if alignment.lower() == "word" or alignment.lower() == "wd":
            json_arr = pd.concat(all_trans)
        else:
            json_arr = pd.DataFrame(all_trans)
            json_arr.columns = ["sid", "recording_id", "utt_num", "speaker", "timestart", "timeend", "utt"]

        return json_arr

    def convert_trs(self):
        """
        Convert trs to a format that is usable in csv file
        :return: the data as a pd dataframe
        """
        trs_arr = []
        with open(self.tfile, "r") as trs:
            if len(self.tfile.split(" ")) > 1:
                print(self.tfile)
                participant = self.tfile.split(" ")[0]
                recording_id = self.tfile.split(" ")[1].split(".trs")[0]
            else:
                participant = "TODO"
                recording_id = self.tfile.split(".trs")[0]

            # for line in trs:
            # try:
            utt = 0
            spkr = 0
            wd_num = 0
            coach_participant = {}

            prev_endtime = 0

            for line in trs:
                if "<Speaker id" in line:
                    participant_num = re.search(r'"spkr(\d)', line).group(1)
                    participant = re.search(r'name="(\S+)"', line).group(1)
                    coach_participant[participant_num] = participant
                if "<Turn " in line:
                    # print(line)
                    timestart = re.search(r'startTime="(\d+.\d+)"', line).group(1)
                    timeend = re.search(r'endTime="(\d+.\d+)"', line).group(1)
                    speaker = re.search(r'spkr(\d)">', line).group(1)
                    word = re.search(r"/>(\S+)</Turn>", line).group(1)

                    if (
                            coach_participant[speaker] == "coach"
                            or coach_participant[speaker] == "Coach"
                    ):
                        # print("Coach found")
                        real_speaker = 2
                    elif (
                            coach_participant[speaker] == "participant"
                            or coach_participant == "Participant"
                    ):
                        # print("Participant found")
                        real_speaker = 1
                    else:
                        print("There was some problem here")
                        print(self.tfile)
                        print(coach_participant[speaker])
                        sys.exit(1)

                    wd_num += 1
                    if spkr != real_speaker:
                        spkr = real_speaker
                        utt += 1
                    # CHANGE UTTERANCE IF PAUSE OF 500MS OR GREATER
                    elif float(timestart) - float(prev_endtime) > .5:
                        utt += 1

                    prev_endtime = timeend

                    trs_arr.append(
                        [participant, recording_id, real_speaker, timestart, timeend, word, utt, wd_num]
                    )
        # convert to pandas dataframe
        trs_arr = pd.DataFrame(trs_arr)
        trs_arr.columns = ["sid", "recording_id", "speaker", "timestart", "timeend", "word", "utt_num", "word_num"]

        return trs_arr

    def flatten_data(self, word_level_data):
        """
        If data should be utterance, aligned, this collapses
        it into one utterance per line
        :return:
        """

        colnames = ['sid', 'recording_id', 'utt_num', 'speaker', 'timestart', 'timeend', 'utt']

        all_utts = []

        for i in (word_level_data['utt_num'].unique()):
            utt = word_level_data.query(f'utt_num=={i}')

            all_utts.append([utt['sid'].iloc[0],
                             utt['recording_id'].iloc[0],
                             i,
                             utt['speaker'].iloc[0],
                             utt['timestart'].iloc[0],
                             utt['timeend'].iloc[-1],
                             utt['word'].str.cat(sep=" ")
                             ])

        utt_level_data = pd.DataFrame(all_utts, columns=colnames)

        return utt_level_data

    def save_data(self, data, savepath):
        """
        To save the data that has been converted
        (and possibly flattened) as a csv file
        :return:
        """
        path_to_saver = savepath.rsplit("/", 1)[0]
        # make sure the full save path exists; if not, create it
        os.system(
            f'if [ ! -d "{path_to_saver}" ]; then mkdir -p {path_to_saver}; fi'
        )

        if os.path.exists(savepath):
            if os.path.getsize(savepath) == 0:
                # if csv exists but is empty
                # include header
                data.to_csv(savepath, index=False)
            else:
                # append this to csv of already-existing files
                data.to_csv(savepath, mode='a', index=False, header=False)
        else:
            # if csv does not exist, make it + include header
            data.to_csv(savepath, index=False)


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


if __name__ == "__main__":
    cpath = "../../lives_test/to_do"

    # preprocess_lives(cpath, flatten_data=True, json_data=True)
    preprocess_lives(cpath, flatten_data=False, json_data=True)