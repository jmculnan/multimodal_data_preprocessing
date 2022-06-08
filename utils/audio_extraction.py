# extract information from audio
# this information is modified from the code in:
# https://github.com/jmculnan/audio_feature_extraction (private repo)

# required packages
import os, sys
import json, re
from pprint import pprint
import subprocess as sp
from tqdm import tqdm

import pandas as pd

# from wav2vec.create_spectrograms_with_wav2vec import get_and_save_spectrogram


class TRSToCSV:
    """
    Takes a trs file and converts it to a csv
    lines of csvfile are speaker,timestart,timeend, word, utt_num where:
        speaker   = caller or patient
        timestart = time of start of word
        timeend   = time of end of word
        word      = the word predicted
        utt_num   = the utterance number within the conversation
        word_num  = the word number (useful for averaging over words)
    """

    def __init__(self, path, trsfile):
        self.path = path
        self.tname = trsfile
        self.tfile = f"{path}/{trsfile}.trs"

    def convert_trs(self, savepath):
        trs_arr = [["speaker", "timestart", "timeend", "word", "utt_num", "word_num"]]
        with open(self.tfile, "r") as trs:
            print(self.tfile)
            # for line in trs:
            # try:
            utt = 0
            spkr = 0
            wd_num = 0
            coach_participant = {}
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

                    if coach_participant[speaker].lower() == "coach":
                        # print("Coach found")
                        real_speaker = 2
                    elif coach_participant[speaker].lower() == "participant":
                        # print("Participant found")
                        real_speaker = 1
                    else:
                        exit("There was some problem here")

                    wd_num += 1
                    if spkr != real_speaker:
                        spkr = real_speaker
                        utt += 1

                    trs_arr.append(
                        [real_speaker, timestart, timeend, word, utt, wd_num]
                    )
        with open(f"{savepath}/{self.tname}.tsv", "w") as cfile:
            for item in trs_arr:
                cfile.write("\t".join([str(x) for x in item]) + "\n")


class ExtractAudio:
    """
    Takes audio and extracts features from it using openSMILE
    """

    def __init__(self, path, audiofile, savedir, smilepath="~/opensmile-3.0"):
        self.path = path
        self.afile = path + "/" + audiofile
        self.savedir = savedir
        self.smile = smilepath

    def save_acoustic_csv(self, feature_set, savename):
        """
        Get the CSV for set of acoustic features for a .wav file
        feature_set : the feature set to be used
        savename : the name of the saved CSV
        Saves the CSV file
        """
        conf_dict = {
            "ISO9": "is09-13/IS09_emotion.conf",
            "IS10": "is09-13/IS10_paraling.conf",
            "IS12": "is09-13/IS12_speaker_trait.conf",
            "IS13": "is09-13/IS13_ComParE.conf",
        }

        fconf = conf_dict.get(feature_set, "IS13_ComParE.conf")

        # check to see if save path exists; if not, make it
        os.makedirs(self.savedir, exist_ok=True)

        # run openSMILE
        sp.run(
            [
                f"{self.smile}/build/progsrc/smilextract/SMILExtract",
                "-C",
                f"{self.smile}/config/{fconf}",
                "-I",
                self.afile,
                "-lldcsvoutput",
                f"{self.savedir}/{savename}",
            ]
        )


class AudioSplit:
    """ Takes audio, can split and join using ffmpeg"""
    def __init__(self, base_path, audio_name, save_ext=None):
        self.path = base_path,
        self.fname = audio_name
        if not audio_name.endswith(".wav"):
            self.fpath = f"{base_path}/{audio_name}.wav"
        else:
            self.fpath = f"{base_path}/{audio_name}"

        if save_ext is not None:
            self.savepath = f"{base_path}/{save_ext}"
        else:
            self.savepath = base_path

    def split_audio_with_pandas(self, utt_df):
        """
        Split audio file based on input pandas df
        Df contains columns ['speaker'], ['startTime'], ['endTime']
        :return:
        """

        os.makedirs(self.savepath, exist_ok=True)

        for row in tqdm(utt_df.itertuples(), total=len(utt_df)):
            # print(row)
            speaker = row.speaker
            recording_id = row.recording_id
            utt_num = row.utt_num
            timestart = row.timestart
            timeend = row.timeend

            sp.run(
                [
                    "ffmpeg",
                    "-i",
                    self.fpath,
                    "-ss",
                    str(timestart),
                    "-to",
                    str(timeend),
                    f"{self.savepath}/{recording_id}_utt{utt_num}_speaker{speaker}.wav",
                    "-loglevel",
                    "quiet",
                ]
            )


# class AudioSplit:
#     """Takes audio, can split and join using ffmpeg"""
#
#     def __init__(self, path, pathext, audio_name, diarized_csv):
#         self.path = path
#         self.aname = audio_name
#         self.cname = diarized_csv
#         self.afile = f"{path}/{audio_name}"
#         self.cfile = f"{path}/{diarized_csv}"
#         self.ext = pathext
#         self.full_path = f"{path}/{pathext}"
#
#     def split_audio(self):
#         """
#         Splits audio based on an input csvfile.
#         csvfile is assumed to start with the following format:
#           speaker,timestart,timeend where
#           speaker   = caller or patient
#           timestart = time of start of turn
#           timeend   = time of turn end
#         """
#
#         os.makedirs(self.full_path, exist_ok=True)
#
#         with open(self.cfile, "r") as csvfile:
#             for n, line in enumerate(csvfile):
#                 speaker, timestart, timeend = line.strip().split(",")[:3]
#                 os.makedirs(f"{self.full_path}/{speaker}", exist_ok=True)
#                 sp.run(
#                     [
#                         "ffmpeg",
#                         "-i",
#                         self.afile,
#                         "-ss",
#                         str(timestart),
#                         "-to",
#                         str(timeend),
#                         f"{self.full_path}/{speaker}/{n}",
#                         "-loglevel",
#                         "quiet",
#                     ]
#                 )
#                 # Consider using a tqdm progress bar here - Adarsh
#                 if n % 1000 == 0:
#                     print(f"Completed {n + 1} lines")
#
#     def make_textfile(self, audiodir, speaker):
#         """
#         Make a .txt file containing the names of all audio in the directory
#         Used for ffmpeg concatenation
#         """
#         txtfilepath = f"{self.full_path}/{speaker}/{self.ext}-{speaker}.txt"
#         with open(txtfilepath, "w") as txtfile:
#             for item in os.listdir(audiodir):
#                 if item[-4:] == ".wav":
#                     txtfile.write(f"file '{item}'\n")
#
#     def join_audio(self, txtfile, speaker):
#         """
#         Joins audio in an input directory using a textfile with path info
#         """
#         os.makedirs(f"{self.path}/output", exist_ok=True)
#
#         outputname = f"{self.ext}-{speaker}.wav"
#
#         sp.run(
#             [
#                 "ffmpeg",
#                 "-f",
#                 "concat",
#                 "-safe",
#                 "0",
#                 "-i",
#                 f"{self.full_path}/{speaker}/{txtfile}",
#                 "-c",
#                 "copy",
#                 f"{self.path}/output/{outputname}",
#                 "-loglevel",
#                 "quiet",
#             ]
#         )
#         print(f"Concatenation completed for {self.full_path}")
#

class GetFeatures:
    """
    Takes input files and gets acoustic features
    Organizes features as required for this project
    Combines data from acoustic csv + transcription csv
    """

    def __init__(self, path, acoustic_csv, trscsv):
        self.path = path
        self.acoustic_csv = acoustic_csv
        self.trscsv = trscsv

    def get_features_dict(self, dropped_cols=None):
        """
        Get the set of phonological/phonetic features
        """
        # create a holder for features
        feature_set = {}

        # iterate through csv files created by openSMILE
        for csvfile in os.listdir(self.savepath):
            if csvfile.endswith(".csv"):
                csv_name = csvfile.split(".")[0]
                # get data from these files
                csv_data = pd.read_csv(f"{self.savepath}/{csvfile}", sep=";")
                # drop name and time frame, as these aren't useful
                if dropped_cols:
                    csv_data = self.drop_cols(csv_data, dropped_cols)
                else:
                    csv_data = (
                        csv_data.drop(["name", "frameTime"], axis=1).to_numpy().tolist()
                    )
                if "nan" in csv_data or "NaN" in csv_data or "inf" in csv_data:
                    pprint(csv_data)
                    print("Data contains problematic data points")
                    sys.exit(1)

                # add it to the set of features
                feature_set[csv_name] = csv_data

        return feature_set


# def run_feature_extraction(audio_path, feature_set, save_dir):
#     """
#     Run feature extraction from audio_extraction.py for a dataset
#     """
#     # make sure the full save path exists; if not, create it
#     os.system(f'if [ ! -d "{save_dir}" ]; then mkdir -p {save_dir}; fi')
#
#     # save all files in the directory
#     for wfile in os.listdir(audio_path):
#         save_name = str(wfile.split(".wav")[0]) + f"_{feature_set}.csv"
#         audio_extractor = ExtractAudio(
#             audio_path, wfile, save_dir, "../../opensmile-3.0"
#         )
#         audio_extractor.save_acoustic_csv(feature_set, save_name)


def run_feature_extraction(audio_path, feature_set, save_dir):
    """
    Run feature extraction from audio_extraction.py for a dataset
    :param audio_path: the full path to the directory containing audio files
    :param feature_set: the feature set used;
        For openSMILE, the feature set is one of IS09-IS13
        For spectrograms, feature set is `spectrogram` or `spec`
        For ASR-preprocessed feats, feature set is `asr` or `wav2vec`
    """
    # make sure the full save path exists; if not, create it
    os.system(f'if [ ! -d "{save_dir}" ]; then mkdir -p {save_dir}; fi')

    # save all files in the directory
    for wfile in tqdm(os.listdir(audio_path), desc=f"Processing files in {audio_path}"):
        if wfile.endswith(".wav"):
            save_name = str(wfile.split(".wav")[0]) + f"_{feature_set}.csv"
            if feature_set.lower() in ["is09", "is10", "is11", "is12", "is13"]:
                audio_extractor = ExtractAudio(
                    audio_path, wfile, save_dir, "../../opensmile-3.0"
                )
                audio_extractor.save_acoustic_csv(feature_set, save_name)
            # todo: reimplement me
            # elif feature_set.lower() in ["spectrogram", "spec"]:
            #     get_and_save_spectrogram(audio_path, wfile, save_dir, resample=None)
            elif feature_set.lower() in ["asr", "wav2vec"]:
                pass


def transform_audio(txtfile):
    """
    Used for taking audio and transforming it in the way initially envisioned
    for LIvES project.
    txtfile = the path to a file containing rows of:
        path : a path to the audio data
        trsfile : the name of the transcription file (without .trs)
        callwav : the name of wav file being transformed
    todo: does this need to be changed to be relevant for this input?
    """
    with open(txtfile, "r") as tfile:
        # print("textfile opened!")
        for line in tfile:
            # print("Line is: " + line)
            path, trsfile, callwav = line.strip().split(",")

            csvfile = f"{trsfile}.csv"
            extension = callwav.split(".")[0]
            speakers = ["1", "2"]

            # diarized_input = DiarizedToCSV(path, jsonfile)
            diarized_input = TRSToCSV(path, trsfile)
            # print("diarized_input created")
            diarized_input.convert_trs()
            # print("conversion to json happened")

            audio_input = AudioSplit(path, extension, callwav, csvfile)
            audio_input.split_audio()

            for speaker in speakers:
                audio_input.make_textfile(f"{path}/{extension}/{speaker}", speaker)
                audio_input.join_audio(f"{extension}-{speaker}.txt", speaker)

            sp.run(["rm", "-r", f"{path}/{extension}"])


def expand_words(trscsv, file_to_save):
    """
    Expands transcription file to include values at every 10ms
    Used to combine word, speaker, utt information with features
    extracted from OpenSMILE
    :param trscsv: the transcription tsv
    :param file_to_save:
    :return:
    """
    saver = [["frameTime", "speaker", "word", "utt_num", "word_num"]]
    with open(trscsv, "r") as tcsv:
        tcsv.readline()
        for line in tcsv:
            (speaker, timestart, timeend, word, utt_num, wd_num,) = line.strip().split(
                "\t"
            )
            saver.append([timestart, speaker, word, utt_num, wd_num])
            newtime = float(timestart) + 0.01
            while newtime < float(timeend):
                newtime += 0.01
                saver.append([str(newtime), speaker, word, utt_num, wd_num])
            saver.append([timeend, speaker, word, utt_num, wd_num])
    with open(file_to_save, "w") as wfile:
        for item in saver:
            wfile.write("\t".join(item) + "\n")


def avg_feats_across_words(feature_df):
    """
    Takes a pandas df of acoustic feats and collapses it into one
    with feats avg'd across words
    :param feature_df: pandas dataframe of features in 24msec intervals
    :return: a new pandas df
    """
    # summarize + avg like dplyr in R
    feature_df = feature_df.groupby(
        ["word", "speaker", "utt_num", "word_num"], sort=False
    ).mean()
    feature_df = feature_df.reset_index()
    return feature_df


def convert_to_wav(nonwav_file):
    """
    convert audio from another format to wav
    nonwav_file : the name of the original file
    ext:
    """
    # check for valid extension
    if nonwav_file.endswith(".m4a") or nonwav_file.endswith(".mp4"):
        # grab name without extension
        wav_name = f"{nonwav_file[:-4]}.wav"
    elif nonwav_file.endswith(".flac"):
        # get name without extension
        wav_name = f"{nonwav_file[:-5]}.wav"
    else:
        exit(f"{nonwav_file} is an unsupported file type")

    if not os.path.exists(wav_name):
        sp.run(["ffmpeg", "-i", nonwav_file, "-ac", "1", wav_name])
    else:
        print(f"{wav_name} already exists")

    return wav_name


def extract_portions_of_mp4_or_wav(
    path_to_sound_file,
    sound_file,
    start_time,
    end_time,
    save_path=None,
    short_file_name=None,
):
    """
    Extracts only necessary portions of a sound file
    sound_file : the name of the full file to be adjusted
    start_time : the time at which the extracted segment should start
    end_time : the time at which the extracted segment should end
    short_file_name : the name of the saved short sound file
    """
    # set full path to file
    full_sound_path = os.path.join(path_to_sound_file, sound_file)

    # check sound file extension
    if sound_file.endswith(".mp4") or sound_file.endswith(".m4a"):
        full_sound_path = convert_to_wav(full_sound_path)

    if not short_file_name:
        print("short file name not found")
        short_file_name = f"{sound_file.split('.')[0]}_{start_time}_{end_time}.wav"

    if save_path is not None:
        save_name = f"{save_path}/{short_file_name}"
    else:
        save_name = f"{path_to_sound_file}/{short_file_name}"

    # make sure the full save path exists; if not, create it
    os.system(f'if [ ! -d "{save_path}" ]; then mkdir -p {save_path}; fi')

    # get shortened version of file
    sp.run(
        [
            "ffmpeg",
            "-i",
            full_sound_path,
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            str(save_name),
        ]
    )

    return save_name
