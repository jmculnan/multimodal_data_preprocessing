# see if we can create a dataset-agnostic data prep class
import math
import glob
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

        # get acoustic dictionary

        # get acoustic set

        # get data tensors

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
        where data is the acoustic feature tensor for that item
    :param file_path: the path to the dir containing data files
    :param dataset: the dataset (e.g. meld, firstimpr)
    :param feature_set: the set used (IS09-13)
    :param use_cols: whether to select specific columns
    :return: dict of id : data pairs; length of each data point
    """
    acoustic_dict = {}
    # todo: is this the right type for this?
    acoustic_lengths = {}

    # get the acoustic features files
    for feats_file in glob.glob(f"{file_path}/{feature_set}/*_{feature_set}.csv"):

        # read each file as a pandas df
        if use_cols is not None:
            feats = pd.read_csv(
                f"{file_path}/{feature_set}/{feats_file}", usecols=use_cols, sep=";"
            )
        else:
            feats = pd.read_csv(f"{file_path}/{feature_set}/{feats_file}", sep=";")
            feats.drop(["name", "frameTime"], axis=1, inplace=True)

        # get the id
        if dataset == "meld":
            dia_id, utt_id = feats_file.split("_")[:2]
            id = (dia_id, utt_id)
        else:
            id = feats_file.split(f"_{feature_set}.csv")[0]

        # save the dataframe to a dict with id as key
        if feats.shape[0] > 0:
            # todo: should we convert to torch tensor instead?
            acoustic_dict[id] = feats.values.tolist()
            # do this so we can ensure same order of lengths and feats
            acoustic_lengths[id] = feats.shape[0]

        # delete the features df bc it takes up lots of space
        del feats

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
    Prepare the acoustic data using the acoustic dict
    :param file_path: the path to dir containing data files
    :param text_file: name of file with utterances + labels
    :param dataset: the dataset
    :param feature_set: the feature set used (IS09-13)
    :param longest_acoustic:
    :param add_avging:
    :return:
    """
    # get acoustic dict and lengths
    acoustic_dict, acoustic_lengths = make_acoustic_dict(file_path, feature_set, use_cols)

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
                # set an intermediate holder
                acoustic_holder = torch.zeros((longest_acoustic, acoustic_lengths[item]))

                # add acoustic features to holder of features
                for i, feats in enumerate(acoustic_data):
                    if i >= longest_acoustic:
                        break
                    for i, feat in enumerate(feats):
                        acoustic_holder[i][j] = feat 