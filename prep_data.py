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
            utterance_fname,
            use_cols=None
    ):
        # set path to data files
        self.d_type = data_type.lower()
        self.path = data_path

        # set list of acoustic feature columns to extract
        #   or leave as None to extract all
        self.acoustic_cols_used = use_cols

        # set feature set
        self.fset = feature_set

        # set path to train, dev, test
        if self.d_type == "meld" or self.d_type == "firstimpr" or self.d_type == "chalearn":
            train_path, dev_path, test_path = get_paths(self.d_type, data_path)
            self.paths = {"train": train_path, "dev": dev_path, "test": test_path}
        else:
            self.paths = {"all": data_path}

        # get text + gold files
        # todo: verify sep is always \t
        try:
            train_data = pd.read_csv(f"{self.paths['train']}/{utterance_fname}", sep="\t")
            dev_data = pd.read_csv(f"{self.paths['dev']}/{utterance_fname}", sep="\t")
            test_data = pd.read_csv(f"{self.paths['test']}/{utterance_fname}", sep="\t")
            all_data = pd.concat([train_data, dev_data, test_data], axis=0)
        except KeyError:
            all_data = pd.read_csv(f"{self.paths['all']}/{utterance_fname}", sep="\t")

        # set tokenizer
        self.tokenizer = get_tokenizer("basic_english")

        # get longest utt
        self.longest_utt = get_longest_utt(all_data, self.tokenizer)

        # get acoustic dict and lengths
        train_acoustic_dict, train_acoustic_lengths = make_acoustic_dict(self.paths["train"],
                                                                         self.d_type, self.fset,
                                                                         self.acoustic_cols_used)
        dev_acoustic_dict, dev_acoustic_lengths = make_acoustic_dict(self.paths["dev"],
                                                                         self.d_type, self.fset,
                                                                         self.acoustic_cols_used)
        test_acoustic_dict, test_acoustic_lengths = make_acoustic_dict(self.paths["test"],
                                                                         self.d_type, self.fset,
                                                                         self.acoustic_cols_used)

        # get all used ids
        train_ids = self.get_all_used_ids(train_data, train_acoustic_dict)
        dev_ids = self.get_all_used_ids(dev_data, dev_acoustic_dict)
        test_ids = self.get_all_used_ids(test_data, test_acoustic_dict)

        # get acoustic set for train, dev, test partitions
        (self.train_acoustic,
         self.train_usable_utts,
         self.train_acoustic_lengths
         ) = self.make_acoustic_set(self.paths["train"],
                                    add_avging=False)

        # use acoustic sets to get data tensors

        # get acoustic means

        # combine xs and ys

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
        all_used = set(valid_ids).intersection(set(acoustic_dict.keys()))

        return all_used



    def make_acoustic_set(
            self,
            acoustic_dict,
            longest_acoustic=1500,
            add_avging=True,
    ):
        """
        Prepare the acoustic data
        Includes creation of acoustic dict
        :param file_path: the path to dir containing acoustic files
        :param used_ids: a list of all unique ids connected with data
        :param longest_acoustic:
        :param add_avging:
        :param use_cols: list of specific column names to select
        :return:
        """

        # set holders for acoustic data
        all_acoustic = []
        all_ids = []

        # for all items with audio + gold label
        for item in valid_ids:
            # if the item has an acoustic feats file
            if item in acoustic_dict.keys():
                # add this item to list of usable utterances
                all_ids.append(item)

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

        return all_acoustic, all_ids, acoustic_lengths

    def make_data_tensors(self, text_data, usable_utts, glove):
        """
        Prepare tensors of utterances, genders, gold labels
        :param text_data: the df containing the text, gold, etc
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

        if self.d_type == "meld":
            pass

        for idx, row in text_data.iterrows():

            pass


def get_longest_utt(data, tokenizer):
    """
    Get the length of longest utterance in the dataset
    :param data: a pandas df containing all utterances
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




if __name__ == "__main__":
    chalearn = DataPrep("chalearn", "../../datasets/multimodal_datasets/Chalearn",
                        "IS10", "gold_and_utts.tsv")