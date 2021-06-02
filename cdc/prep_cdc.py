# prepare the columbia deception corpus
import os
import glob
import re
import pandas as pd
from utils.data_prep_helpers import split_string_time


def preprocess_cdc(
        corpus_path,
        save_filename,
        delete_existing_data_file=True
):
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
            prep_cdc_trs_data(f, ltf_file, f"{corpus_path}/{save_filename}")


def prep_cdc_trs_data(trs_file_path, ltf_file_path, save_filename_and_path):
    """
    Convert CDC trs files to organized tsv
    """
    # holder for utterance and gold data
    all_utts = []

    # add header if the file doesn't exist
    if not os.path.exists(save_filename_and_path):
        all_utts.append("speaker\tutt\ttime_start\ttime_end\ttruth_value")

    # prepare the ltf (gold labeled) data
    ltf_data = []
    with open(ltf_file_path, 'r') as ltf_file:
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
                    start_time = float(re.search(r'<Sync time="(\S+)"/>', line).group(1))

                    try:
                        # get end time -- should usually be two lines after
                        end_time = float(re.search(r'<Sync time="(\S+)"/>', used_lines[i + 2]).group(1))
                    # if end time isn't where we expect it
                    except AttributeError:
                        end_found = False
                        j = 1
                        # search for end time
                        while not end_found:
                            if used_lines[i + 2 + j].startswith("<Sync time="):
                                end_time = float(re.search(r'<Sync time="(\S+)"/>', used_lines[i + 2 + j]).group(1))
                                end_found = True
                            else:
                                j += 1

                    # match start and end times with truth values
                    truth_value = None
                    find_truth = True
                    # find overlap between gold label times and speech times
                    while find_truth:
                        # see if gold label time fully contains speech times
                        if ltf_data[ltf_pointer][1] <= start_time and ltf_data[ltf_pointer][2] >= end_time:
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

                    all_utts.append(f"{speaker}\t{utt}\t{start_time}\t{end_time}\t{truth_value}")

    # save the data
    with open(save_filename_and_path, "a") as savefile:
        savefile.write("\n".join(all_utts))
        savefile.write("\n")


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
        "/"
    ]
    unusable = "|".join(unusable)
    string = string.strip()
    string = re.sub(unusable, "", string)
    string = re.sub("&lt;UNIN&gt;", "<UNK>", string)
    string = re.sub(r"%[A-Z]+%+\S+%", "", string)

    return string


if __name__ == "__main__":

    # fpath = "../../datasets/multimodal_datasets/Columbia_deception_corpus/data/S-1A/S-1A_R_16k.punc.trs"
    # ltf_path = "../../datasets/multimodal_datasets/Columbia_deception_corpus/data/S-1A/S-1A.ltf"
    # savepath = "../../datasets/multimodal_datasets/Columbia_deception_corpus/DELETEME.tsv"
    #
    # prep_cdc_trs_data(fpath, ltf_path, savepath)

    corpus_path = "../../datasets/multimodal_datasets/Columbia_deception_corpus"
    save_filename = "cdc_gold.tsv"

    preprocess_cdc(corpus_path, save_filename, delete_existing_data_file=True)