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

from combine_xs_and_ys_by_dataset import combine_xs_and_ys_chalearn
from utils.audio_extraction import (ExtractAudio,
                                    convert_to_wav,
                                    run_feature_extraction
                                    )
from make_data_tensors_by_dataset import *

import pandas as pd

from utils.data_prep_helpers import (
    get_class_weights,
    get_gender_avgs,
    clean_up_word,
    get_max_num_acoustic_frames,
    transform_acoustic_item,
    get_acoustic_means
    )


class StandardPrep:
    """
    Prep a dataset with train, dev, test partitions
    """
    def __init__(
            self,
            data_type,
            data_path,
            feature_set,
            utterance_fname,
            glove,
            use_cols=None,
    ):
        # set path to data files
        self.d_type = data_type.lower()
        self.path = data_path

        # set list of acoustic feature columns to extract
        #   or leave as None to extract all
        self.acoustic_cols_used = use_cols

        # set feature set
        self.fset = feature_set

        # set train, dev, test paths
        train_path, dev_path, test_path = get_paths(self.d_type, data_path)
        self.paths = {"train": train_path, "dev": dev_path, "test": test_path}

        # read in each partition
        train_data = pd.read_csv(f"{self.paths['train']}/{utterance_fname}", sep="\t")
        dev_data = pd.read_csv(f"{self.paths['dev']}/{utterance_fname}", sep="\t")
        test_data = pd.read_csv(f"{self.paths['test']}/{utterance_fname}", sep="\t")
        all_data = pd.concat([train_data, dev_data, test_data], axis=0)

        # set tokenizer
        self.tokenizer = get_tokenizer("basic_english")

        # get longest utt
        self.longest_utt = get_longest_utt(all_data, self.tokenizer)

        # set longest accepted acoustic file
        self.longest_acoustic = 1500

        # set DataPrep instance for each partition
        self.train_prep = DataPrep(self.d_type, train_data, self.paths['train'],
                                   self.tokenizer, self.fset,
                                   self.acoustic_cols_used, self.longest_utt,
                                   self.longest_acoustic, glove, "train")
        self.dev_prep = DataPrep(self.d_type, dev_data, self.paths['dev'],
                                 self.tokenizer, self.fset,
                                 self.acoustic_cols_used, self.longest_utt,
                                 self.longest_acoustic, glove, "dev")
        self.test_prep = DataPrep(self.d_type, test_data, self.paths['test'],
                                  self.tokenizer, self.fset,
                                  self.acoustic_cols_used, self.longest_utt,
                                  self.longest_acoustic, glove, "test")


class DataPrep:
    """
    A class to prepare datasets
    Should allow input from meld, firstimpr, mustard, ravdess, cdc
    """
    def __init__(
            self,
            data_type,
            data,
            data_path,
            tokenizer,
            feature_set,
            acoustic_cols_used,
            longest_utt,
            longest_acoustic,
            glove,
            partition
    ):
        # set data type
        self.d_type = data_type

        # set tokenizer
        self.tokenizer = tokenizer

        # get acoustic dict and lengths
        self.acoustic_dict, self.acoustic_lengths = make_acoustic_dict(data_path,
                                                                       data_type,
                                                                       feature_set,
                                                                       acoustic_cols_used)

        # get all used ids
        self.used_ids = self.get_all_used_ids(data, self.acoustic_dict)

        # get acoustic set for train, dev, test partitions
        self.acoustic_tensor = self.make_acoustic_set(self.acoustic_dict,
                                                      longest_acoustic,
                                                      add_avging=False)

        # use acoustic sets to get data tensors
        self.data_tensors = self.make_data_tensors(data, longest_utt, glove)

        # get acoustic means
        if partition == "train":
            self.acoustic_means, self.acoustic_stdev = get_acoustic_means(self.acoustic_tensor)

    def combine_xs_and_ys(self):
        """
        Combine the xs and y data
        :return: all data as list of tuples of tensors
        """
        if self.d_type == "meld":
            pass
        elif self.d_type == "mustard":
            pass
        elif self.d_type == "chalearn" or self.d_type == "firstimpr":
            combine_xs_and_ys_chalearn(self.acoustic_tensor, self.acoustic_dict,
                                       self.acoustic_means, self.acoustic_stdev)
        elif self.d_type == "ravdess":
            pass
        elif self.d_type == "cdc":
            pass

    def get_all_used_ids(self, text_data, acoustic_dict):
        """
        Get a list of all the ids that have both acoustic and text/gold info
        :return: array of all valid ids
        """
        # get list of valid dialogues/utterances
        if self.d_type == "meld":
            valid_ids = text_data["DiaID_UttID"].tolist()
            valid_ids = [(item.split("_")[0], item.split("_")[1]) for item in valid_ids]
        elif self.d_type == "mustard":
            valid_ids = text_data["clip_id"].tolist()
        elif self.d_type == "chalearn" or self.d_type == "firstimpr":
            valid_ids = text_data["file"].tolist()
            valid_ids = [item.split(".mp4")[0] for item in valid_ids]
        elif self.d_type == "ravdess":
            pass
        elif self.d_type == "cdc":
            pass

        # get intersection of valid ids and ids present in acoustic data
        all_used_ids = set(valid_ids).intersection(set(acoustic_dict.keys()))

        return all_used_ids

    def make_acoustic_set(
            self,
            acoustic_dict,
            longest_acoustic=1500,
            add_avging=True,
    ):
        """
        Prepare the acoustic data
        Includes creation of acoustic dict
        :param acoustic_dict: a dict of acoustic feat dfs
        :param longest_acoustic: the longest allowed acoustic df
        :param add_avging: whether to average features
        :return:
        """

        # set holders for acoustic data
        all_acoustic = []

        # for all items with audio + gold label
        for item in self.used_ids:

            # pull out the acoustic feats df
            acoustic_data = acoustic_dict[item]

            if not add_avging:
                acoustic_data = acoustic_data[acoustic_data.index <= longest_acoustic]
                acoustic_holder = torch.tensor(acoustic_data.values)
            else:
                data_len = len(acoustic_data)
                acoustic_holder = torch.mean(
                    torch.tensor(acoustic_data.values)[math.floor(data_len * 0.25):math.ceil(data_len * 0.75)],
                    dim=0)

            # add features as tensor to acoustic data
            all_acoustic.append(acoustic_holder)

        # delete acoustic dict to save space
        del acoustic_dict

        # pad the sequence and reshape it to proper format
        # this is here to keep the formatting for acoustic RNN
        all_acoustic = nn.utils.rnn.pad_sequence(all_acoustic)
        all_acoustic = all_acoustic.transpose(0, 1)

        print(f"Acoustic set made at {datetime.datetime.now()}")

        return all_acoustic

    def make_data_tensors(self, text_data, longest_utt, glove):
        """
        Prepare tensors of utterances, genders, gold labels
        :param text_data: the df containing the text, gold, etc
        :param longest_utt: length of longest utterance
        :param glove: an instance of class Glove
        :return:
        """
        if self.d_type == "meld":
            data_tensor_dict = make_data_tensors_meld(text_data, longest_utt, glove)
        elif self.d_type == "mustard":
            data_tensor_dict = make_data_tensors_mustard(text_data, longest_utt, glove)
        elif self.d_type == "chalearn" or self.d_type == "firstimpr":
            data_tensor_dict = make_data_tensors_chalearn(text_data, longest_utt, glove)
        elif self.d_type == "ravdess":
            pass
            #data_tensor_dict = make_data_tensors_ravdess(text_data, longest_utt, glove)
        elif self.d_type == "cdc":
            pass
            #data_tensor_dict = make_data_tensors_cdc(text_data, longest_utt, glove)

        return data_tensor_dict


def get_longest_utt(data, tokenizer):
    """
    Get the length of longest utterance in the dataset
    :param data: a pandas df containing all utterances
    :param tokenizer: a tokenizer
    :return: length of longest utterance
    """
    # todo: move this up if needed in other places
    data.columns = data.columns.str.lower()
    all_utts = data["utterance"].tolist()

    # tokenize, clean up, and count all items in dataset
    item_lens = [len(tokenizer(clean_up_word(str(item)))) for item in all_utts]

    # return longest
    return max(item_lens)


def get_paths(data_type, data_path):
    """
    Get the train, dev, test paths based on the data type
    Used with prepartitioned data (meld, firstimpr)
    :param data_type: the name of the dataset of interest
    :param data_path: the base path to dataset
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



# todo: delete this after done testing
#   this repo should have no 'main' functions
if __name__ == "__main__":
    chalearn = DataPrep("chalearn", "../../datasets/multimodal_datasets/Chalearn",
                        "IS10", "gold_and_utts.tsv")