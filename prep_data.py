# see if we can create a dataset-agnostic data prep class
import math
import glob
import datetime
import os
import pickle
import json
import re
import sys
from collections import OrderedDict

import torch
from torch import nn
from torchtext.data import get_tokenizer

from utils.audio_extraction import (ExtractAudio,
                                    convert_to_wav,
                                    run_feature_extraction
                                    )
import pandas as pd

from utils.data_prep_helpers import (
    get_class_weights,
    get_gender_avgs,
    clean_up_word,
    get_max_num_acoustic_frames,
    transform_acoustic_item,
    get_acoustic_means
    )


class DataPrep:
    """
    A class to prepare datasets
    Should allow input from meld, firstimpr, mustard, ravdess, cdc
    """
    def __init__(
            self,
            data_type,
            data_path,
            feature_set,
            utterance_fname
    ):
        # set path to data files
        self.d_type = data_type.lower()
        self.path = data_path

        # set path to train, dev, test
        if self.d_type == "meld" or self.d_type == "firstimpr" or self.d_type == "chalearn":
            train_path, dev_path, test_path = get_paths(self.d_type, data_path)
            self.paths = {"train": train_path, "dev": dev_path, "test": test_path}
        else:
            self.paths = {"all": data_path}

        # get acoustic set for train, dev, test partitions
        (acoustic_train,
         train_usable_utts,
         acoustic_train_lengths
         ) = make_acoustic_set(self.paths["train"], utterance_fname,
                               self.d_type, feature_set, 76, add_avging=False)

        # use acoustic sets to get data tensors

        # get acoustic means

        # combine xs and ys


def get_paths(data_type, data_path):
    """
    Get the train, dev, test paths based on the data type
    Used with prepartitioned data (meld, firstimpr)
    :param data_type: the name of the dataset of interest
    :return: train_path, dev_path, test_path - string paths
    """
    train = f"{data_path}/train"
    test = f"{data_path}/test"
    if data_type == "meld":
        dev = f"{data_path}/dev"
    else:
        dev = f"{data_path}/val"

    return train, dev, test


def make_acoustic_dict(file_path, dataset, feature_set, use_cols=None):
    """
    Use the path to get a dict of unique_id : data pairs
        And the length of each acoustic df
        where data is the acoustic feature tensor for that item
    :param file_path: the path to the dir containing data files
    :param dataset: the dataset (e.g. meld, firstimpr)
    :param feature_set: the set used (IS09-13)
    :param use_cols: whether to select specific columns
    :return: dict of id : data pairs; length of each data point
    """
    print(f"Starting acoustic dict at {datetime.datetime.now()}")
    acoustic_dict = {}
    # todo: is this the right type for this?
    acoustic_lengths = {}

    # get the acoustic features files
    for feats_file in glob.glob(f"{file_path}/{feature_set}/*_{feature_set}.csv"):
        # read each file as a pandas df
        if use_cols is not None:
            feats = pd.read_csv(feats_file, usecols=use_cols, sep=";"
            )
        else:
            feats = pd.read_csv(feats_file, sep=";")
            feats.drop(["name", "frameTime"], axis=1, inplace=True)

        # get the id
        feats_file_name = feats_file.split("/")[-1]

        if dataset == "meld":
            dia_id, utt_id = feats_file_name.split("_")[:2]
            id = (dia_id, utt_id)
        else:
            id = feats_file_name.split(f"_{feature_set}.csv")[0]

        # save the dataframe to a dict with id as key
        if feats.shape[0] > 0:
            # todo: should we convert to torch tensor instead?
            acoustic_dict[id] = feats
            # do this so we can ensure same order of lengths and feats
            acoustic_lengths[id] = feats.shape[0]

        # delete the features df bc it takes up lots of space
        del feats

    print(f"Acoustic dict made at {datetime.datetime.now()}")
    print(f"Len of dict: {len(acoustic_dict.keys())}")
    return acoustic_dict, acoustic_lengths


def make_acoustic_set(
        file_path,
        text_file,
        dataset,
        feature_set,
        longest_acoustic=1500,
        add_avging=True,
        use_cols=None
):
    """
    Prepare the acoustic data
    Includes creation of acoustic dict
    :param file_path: the path to dir containing data files
    :param text_file: name of file with utterances + labels
    :param dataset: the dataset
    :param feature_set: the feature set used (IS09-13)
    :param longest_acoustic:
    :param add_avging:
    :return:
    """
    # get acoustic dict and lengths
    acoustic_dict, acoustic_lengths = make_acoustic_dict(file_path, dataset, feature_set, use_cols)

    # get text
    # todo: verify sep is always \t
    text_data = pd.read_csv(f"{file_path}/{text_file}", sep="\t")

    # get list of valid dialogues/utterances
    if dataset == "meld":
        valid_ids = text_data["DiaID_UttID"].tolist()
        valid_ids = [(item.split("_")[0], item.split("_")[1]) for item in valid_ids]
    elif dataset == "mustard":
        valid_ids = text_data["clip_id"].tolist()
    elif dataset == "chalearn" or dataset == "firstimpr":
        valid_ids = text_data["file"].tolist()
        valid_ids = [item.split(".mp4")[0] for item in valid_ids]
    elif dataset == "ravdess":
        pass
    elif dataset == "cdc":
        pass

    # set holders for acoustic data
    all_acoustic = []
    usable_utts = []

    # for all items with audio + gold label
    for item in valid_ids:
        # if the item has an acoustic feats file
        if item in acoustic_dict.keys():
            # add this item to list of usable utterances
            usable_utts.append(item)

            # pull out the acoustic feats df
            acoustic_data = acoustic_dict[item]

            if not add_avging:
                acoustic_data = acoustic_data[acoustic_data.index <= longest_acoustic]
                acoustic_holder = torch.tensor(acoustic_data.values)
            else:
                data_len = len(acoustic_data)
                acoustic_holder = torch.mean(torch.tensor(acoustic_data.values)[math.floor(data_len * 0.25):math.ceil(data_len * 0.75)], dim=0)

            # add features as tensor to acoustic data
            all_acoustic.append(acoustic_holder)

    # delete acoustic dict to save space
    del acoustic_dict

    # pad the sequence and reshape it to proper format
    # this is here to keep the formatting for acoustic RNN
    all_acoustic = nn.utils.rnn.pad_sequence(all_acoustic)
    all_acoustic = all_acoustic.transpose(0, 1)

    print(f"Acoustic set made at {datetime.datetime.now()}")

    return all_acoustic, usable_utts, acoustic_lengths


def make_data_tensors(text_file, usable_utts, glove):
    """
    Prepare tensors of utterances, genders, gold labels
    :param text_file: the file containing the text
    :param usable_utts: a list of usable utterances
    :param glove: an instance of class Glove
    :return:
    """
    all_utterances = []
    all_speakers = []
    all_genders = []
    all_ids = []
    all_ys = {}

    utt_lengths = []


    pass


if __name__ == "__main__":
    chalearn = DataPrep("chalearn", "../../datasets/multimodal_datasets/Chalearn",
                        "IS10", "gold_and_utts.tsv")