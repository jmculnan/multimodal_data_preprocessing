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
from sklearn.utils import compute_class_weight
from torch import nn
from torchtext.data import get_tokenizer
from tqdm import tqdm

from combine_xs_and_ys_by_dataset import (
    combine_xs_and_ys_firstimpr,
    combine_xs_and_ys_meld,
    combine_xs_and_ys_mustard,
    combine_xs_and_ys_cdc,
    combine_xs_and_ys_mosi, combine_xs_and_ys_lives,
    combine_xs_and_ys_asist
)

from utils.audio_extraction import ExtractAudio, convert_to_wav, run_feature_extraction
from make_data_tensors_by_dataset import *

import pandas as pd

from utils.data_prep_helpers import (
    get_class_weights,
    get_gender_avgs,
    clean_up_word,
    transform_acoustic_item,
    get_acoustic_means,
    create_data_folds,
    get_speaker_to_index_dict,
    create_data_folds_list,
)

from bert.prepare_bert_embeddings import *


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
        glove=None,
        transcription_type="gold",
        use_cols=None,
        avg_acoustic_data=False,
        custom_feats_file=None,
        bert_type="distilbert",
        include_spectrograms=False,
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
        if data_type == "meld" and transcription_type.lower() == "gold":
            sep = ","
        else:
            sep = "\t"
        train_data = pd.read_csv(f"{self.paths['train']}/{utterance_fname}", sep=sep)
        train_data.columns = train_data.columns.str.lower()

        dev_data = pd.read_csv(f"{self.paths['dev']}/{utterance_fname}", sep=sep)
        dev_data.columns = dev_data.columns.str.lower()

        test_data = pd.read_csv(f"{self.paths['test']}/{utterance_fname}", sep=sep)
        test_data.columns = test_data.columns.str.lower()

        all_data = pd.concat([train_data, dev_data, test_data], axis=0)

        # set tokenizer
        if glove is None:
            if bert_type.lower() == "bert":
                self.tokenizer = get_bert_tokenizer()
            else:
                self.tokenizer = get_distilbert_tokenizer()
            self.use_bert = True
        else:
            self.tokenizer = get_tokenizer("basic_english")
            self.use_bert = False

        # get longest utt
        self.longest_utt = get_longest_utt(all_data, self.tokenizer, self.use_bert)

        # set longest accepted acoustic file
        self.longest_acoustic = 1500

        # set DataPrep instance for each partition
        self.train_prep = DataPrep(
            self.d_type,
            train_data,
            self.paths["train"],
            self.tokenizer,
            self.fset,
            self.acoustic_cols_used,
            self.longest_utt,
            self.longest_acoustic,
            glove,
            "train",
            add_avging=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            bert_type=bert_type,
            include_spectrograms=include_spectrograms
        )

        print("deleting train data")
        del train_data

        self.dev_prep = DataPrep(
            self.d_type,
            dev_data,
            self.paths["dev"],
            self.tokenizer,
            self.fset,
            self.acoustic_cols_used,
            self.longest_utt,
            self.longest_acoustic,
            glove,
            "dev",
            add_avging=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            bert_type=bert_type,
            include_spectrograms=include_spectrograms
        )
        self.dev_prep.update_acoustic_means(
            self.train_prep.acoustic_means, self.train_prep.acoustic_stdev
        )

        print("deleting dev data")
        del dev_data

        self.test_prep = DataPrep(
            self.d_type,
            test_data,
            self.paths["test"],
            self.tokenizer,
            self.fset,
            self.acoustic_cols_used,
            self.longest_utt,
            self.longest_acoustic,
            glove,
            "test",
            add_avging=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            bert_type=bert_type,
            include_spectrograms=include_spectrograms
        )
        self.test_prep.update_acoustic_means(
            self.train_prep.acoustic_means, self.train_prep.acoustic_stdev
        )

        del test_data


class SelfSplitPrep:
    """
    A class for when data must be manually partitioned
    """

    def __init__(
        self,
        data_type,
        data_path,
        feature_set,
        utterance_fname,
        glove=None,
        use_cols=None,
        train_prop=0.6,
        test_prop=0.2,
        pred_type=None,
        as_dict=False,
        avg_acoustic_data=False,
        custom_feats_file=None,
        bert_type="distilbert",
        include_spectrograms=False
    ):
        # set path to data files
        self.d_type = data_type.lower()
        self.path = data_path

        # set train and test proportions
        self.train_prop = train_prop
        self.test_prop = test_prop

        # set list of acoustic feature columns to extract
        #   or leave as None to extract all
        self.acoustic_cols_used = use_cols

        # set feature set
        self.fset = feature_set

        # read in data
        if utterance_fname.endswith(".tsv"):
            # most self-split data are TSV
            self.all_data = pd.read_csv(f"{self.path}/{utterance_fname}", sep="\t")
        else:
            # lives data is CSV
            self.all_data = pd.read_csv(f"{self.path}/{utterance_fname}")

        # get dict of all speakers to use
        if (
            data_type == "mustard"
            or data_type == "cdc"
            or data_type == "mosi"
            or data_type == "cmu_mosi"
            or data_type == "cmu-mosi"
        ):
            all_speakers = set(self.all_data["speaker"])
            speaker2idx = get_speaker_to_index_dict(all_speakers)
        elif data_type == "lives":
            # lives data has a more complicated way of getting speaker
            all_speakers = get_lives_speakers(self.all_data)
            speaker2idx = get_speaker_to_index_dict(all_speakers)
        elif data_type == "asist":
            all_speakers = set(self.all_data['participantid'])
            speaker2idx = get_speaker_to_index_dict(all_speakers)
        else:
            speaker2idx = None

        # set tokenizer
        if glove is None:
            if bert_type.lower() == "bert":
                self.tokenizer = get_bert_tokenizer()
            else:
                self.tokenizer = get_distilbert_tokenizer()
            self.use_bert = True
        else:
            self.tokenizer = get_tokenizer("basic_english")
            self.use_bert = False

        # get longest utt
        self.longest_utt = get_longest_utt(self.all_data, self.tokenizer, self.use_bert)

        # set longest accepted acoustic file
        self.longest_acoustic = 1000

        # set DataPrep instance for each partition
        self.train_prep = DataPrep(
            self.d_type,
            self.all_data,
            self.path,
            self.tokenizer,
            self.fset,
            self.acoustic_cols_used,
            self.longest_utt,
            self.longest_acoustic,
            glove,
            "train",
            add_avging=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            bert_type=bert_type,
            include_spectrograms=include_spectrograms
        )

        del self.all_data

        # add pred type if needed (currently just mosi)
        if pred_type is not None:
            self.train_prep.add_pred_type(pred_type)

        self.data = self.train_prep.combine_xs_and_ys(speaker2idx, as_dict)

    def get_data_folds(self):
        train_data, dev_data, test_data = create_data_folds_list(
            self.data, self.train_prop, self.test_prop
        )
        return train_data, dev_data, test_data

def get_updated_class_weights(train_ys):
    """
    Get updated class weights
    Because DataPrep assumes you only enter train set
    :return:
    """
    labels = [int(y) for y in train_ys]
    classes = sorted(list(set(labels)))
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


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
        glove=None,
        partition="train",
        add_avging=False,
        custom_feats_file=None,
        bert_type="distilbert",
        include_spectrograms=False
    ):
        # set data type
        self.d_type = data_type

        # set tokenizer
        self.tokenizer = tokenizer

        # get acoustic dict and lengths
        if custom_feats_file is None:
            self.acoustic_dict, self.acoustic_lengths_dict = make_acoustic_dict(
                data_path, data_type, feature_set, acoustic_cols_used
            )
        else:
            (
                self.acoustic_dict,
                self.acoustic_lengths_dict,
            ) = make_acoustic_dict_with_custom_features(
                data_path, custom_feats_file, data_type, acoustic_cols_used
            )

        # get all used ids
        self.used_ids = self.get_all_used_ids(data, self.acoustic_dict)

        # get acoustic set for train, dev, test partitions
        self.acoustic_tensor, self.acoustic_lengths = self.make_acoustic_set(
            self.acoustic_dict,
            self.acoustic_lengths_dict,
            longest_acoustic,
            add_avging=add_avging,
        )
        del self.acoustic_dict, self.acoustic_lengths_dict

        if include_spectrograms:
            self.spec_dict, self.spec_lengths_dict = make_spectrograms_dict(data_path, data_type)
            self.spec_set, self.spec_lengths_list = self.make_spectrogram_set(self.spec_dict, self.spec_lengths_dict)
            del self.spec_dict, self.spec_lengths_dict
        else:
            self.spec_set = None
            self.spec_lengths_list = None

        # use acoustic sets to get data tensors
        self.data_tensors = self.make_data_tensors(data, longest_utt, glove, bert_type)

        # get acoustic means
        self.acoustic_means = 0
        self.acoustic_stdev = 0
        # if train partition
        if partition == "train":
            # get means and stdev
            self.acoustic_means, self.acoustic_stdev = get_acoustic_means(
                self.acoustic_tensor
            )
            # get class weights
            self.class_weights = get_class_weights(self.data_tensors, self.d_type)

        # add pred type if needed
        self.pred_type = None

    def add_pred_type(self, ptype):
        """
        Add a prediction type for chalearn
        Options: max_class, high-low/binary, high-med-low/ternary
        :param ptype: string name of prediction type
        :return:
        """
        self.pred_type = ptype

    def update_acoustic_means(self, means, stdev):
        """
        If you are not working with train partition, update acoustic means and stdev
            from train to do this correctly
        :return:
        """
        self.acoustic_means = means
        self.acoustic_stdev = stdev

    def combine_xs_and_ys(self, speaker2idx=None, as_dict=False):
        """
        Combine the xs and y data
        :return: all data as list of tuples of tensors
        """
        if self.d_type == "meld":
            combined = combine_xs_and_ys_meld(
                self.data_tensors,
                self.acoustic_tensor,
                self.acoustic_lengths,
                self.acoustic_means,
                self.acoustic_stdev,
                as_dict=as_dict,
                spec_data=self.spec_set,
                spec_lengths=self.spec_lengths_list
            )
        elif self.d_type == "mustard":
            combined = combine_xs_and_ys_mustard(
                self.data_tensors,
                self.acoustic_tensor,
                self.acoustic_lengths,
                self.acoustic_means,
                self.acoustic_stdev,
                speaker2idx,
                as_dict=as_dict,
                spec_data=self.spec_set,
                spec_lengths=self.spec_lengths_list
            )
        elif self.d_type == "chalearn" or self.d_type == "firstimpr":
            combined = combine_xs_and_ys_firstimpr(
                self.data_tensors,
                self.acoustic_tensor,
                self.acoustic_lengths,
                self.acoustic_means,
                self.acoustic_stdev,
                pred_type=self.pred_type,
                as_dict=as_dict,
                spec_data=self.spec_set,
                spec_lengths=self.spec_lengths_list
            )
        elif self.d_type == "cdc":
            combined = combine_xs_and_ys_cdc(
                self.data_tensors,
                self.acoustic_tensor,
                self.acoustic_lengths,
                self.acoustic_means,
                self.acoustic_stdev,
                speaker2idx,
                as_dict=as_dict,
                spec_data=self.spec_set,
                spec_lengths=self.spec_lengths_list
            )
        elif (
            self.d_type == "mosi"
            or self.d_type == "cmu_mosi"
            or self.d_type == "cmu-mosi"
        ):
            combined = combine_xs_and_ys_mosi(
                self.data_tensors,
                self.acoustic_tensor,
                self.acoustic_lengths,
                self.acoustic_means,
                self.acoustic_stdev,
                speaker2idx,
                pred_type=self.pred_type,
                as_dict=as_dict,
                spec_data=self.spec_set,
                spec_lengths=self.spec_lengths_list
            )
        elif self.d_type == "lives":
            combined = combine_xs_and_ys_lives(
                self.data_tensors,
                self.acoustic_tensor,
                self.acoustic_lengths,
                self.acoustic_means,
                self.acoustic_stdev,
                speaker2idx,
                as_dict=as_dict,
                spec_data=self.spec_set,
                spec_lengths=self.spec_lengths_list
            )
        elif self.d_type == "asist":
            combined = combine_xs_and_ys_asist(
                self.data_tensors,
                self.acoustic_tensor,
                self.acoustic_lengths,
                self.acoustic_means,
                self.acoustic_stdev,
                speaker2idx,
                as_dict=as_dict,
                spec_data=self.spec_set,
                spec_lengths=self.spec_lengths_list
            )

        return combined

    def get_all_used_ids(self, text_data, acoustic_dict):
        """
        Get a list of all the ids that have both acoustic and text/gold info
        :return: array of all valid ids
        """
        # get list of valid dialogues/utterances
        if self.d_type == "meld":
            valid_ids = text_data["diaid_uttid"].tolist()
            valid_ids = [(item.split("_")[0], item.split("_")[1]) for item in valid_ids]
        elif self.d_type == "mustard":
            valid_ids = text_data["clip_id"].tolist()
        elif self.d_type == "chalearn" or self.d_type == "firstimpr":
            valid_ids = text_data["file"].tolist()
            valid_ids = [item.split(".mp4")[0] for item in valid_ids]
        elif self.d_type == "cdc":
            valid_ids = text_data["utt_num"].tolist()
            valid_ids = [str(item) for item in valid_ids]
        elif (
            self.d_type == "mosi"
            or self.d_type == "cmu-mosi"
            or self.d_type == "cmu_mosi"
        ):
            valid_ids = text_data["id"].tolist()
        elif self.d_type == "lives":
            text_data['utt_num'] = text_data['utt_num'].astype(str)
            valid_ids = text_data.agg(lambda x: f"{x['recording_id']}_utt{x['utt_num']}_speaker{x['speaker']}", axis=1)
            # valid_ids = text_data[['recording_id', 'utt_num']].agg("_".join, axis=1)
        elif self.d_type == "asist":
            valid_ids = text_data["message_id"].tolist()

        # get intersection of valid ids and ids present in acoustic data
        all_used_ids = set(valid_ids).intersection(set(acoustic_dict.keys()))

        return all_used_ids

    def make_acoustic_set(
        self,
        acoustic_dict,
        acoustic_lengths_dict,
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
        ordered_acoustic_lengths = []

        # for all items with audio + gold label
        for item in tqdm(self.used_ids, desc=f"Preparing acoustic set for {self.d_type}"):

            # pull out the acoustic feats df
            acoustic_data = acoustic_dict[item]
            ordered_acoustic_lengths.append(acoustic_lengths_dict[item])

            if not add_avging:
                acoustic_data = acoustic_data[acoustic_data.index <= longest_acoustic]
                acoustic_holder = torch.tensor(acoustic_data.values)
            else:
                data_len = len(acoustic_data)
                acoustic_holder = torch.mean(
                    torch.tensor(acoustic_data.values)[
                        math.floor(data_len * 0.25) : math.ceil(data_len * 0.75)
                    ],
                    dim=0,
                )

            del acoustic_dict[item]
            del acoustic_lengths_dict[item]

            # add features as tensor to acoustic data
            all_acoustic.append(acoustic_holder)

        # delete acoustic dict to save space
        del acoustic_dict

        # pad the sequence and reshape it to proper format
        # this is here to keep the formatting for acoustic RNN
        all_acoustic = nn.utils.rnn.pad_sequence(all_acoustic)
        all_acoustic = all_acoustic.transpose(0, 1)
        all_acoustic = all_acoustic.float()
        # all_acoustic = all_acoustic.type(torch.FloatTensor)

        print(f"Acoustic set made at {datetime.datetime.now()}")

        return all_acoustic, ordered_acoustic_lengths

    def make_spectrogram_set(
        self,
        spec_dict,
        spec_lengths_dict,
        longest_spec=1500
    ):
        """
        Prepare the spectrogram data
        :param spec_dict: a dict of acoustic feat dfs
        :param spec_lengths_dict: a dict of lengths of spectrograms
        :param longest_spec: the longest allowed spec df
        :return:
        """

        # set holders for acoustic data
        all_spec = []
        ordered_spec_lengths = []

        # for all items with audio + gold label
        for item in tqdm(self.used_ids, desc=f"Preparing spec set for {self.d_type}"):

            # pull out the spec feats df
            spec_data = spec_dict[item]
            ordered_spec_lengths.append(spec_lengths_dict[item])

            spec_data = spec_data[spec_data.index <= longest_spec]
            spec_holder = torch.tensor(spec_data.values)

            # add features as tensor to acoustic data
            all_spec.append(spec_holder)

        # delete acoustic dict to save space
        del spec_dict

        print(f"Acoustic set made at {datetime.datetime.now()}")

        return all_spec, ordered_spec_lengths

    def make_data_tensors(
        self, text_data, longest_utt, glove=None, bert_type="distilbert"
    ):
        """
        Prepare tensors of utterances, genders, gold labels
        :param text_data: the df containing the text, gold, etc
        :param longest_utt: length of longest utterance
        :param glove: an instance of class Glove
        :return:
        """
        if self.d_type == "meld":
            data_tensor_dict = make_data_tensors_meld(
                text_data, self.used_ids, longest_utt, self.tokenizer, glove, bert_type
            )
        elif self.d_type == "mustard":
            data_tensor_dict = make_data_tensors_mustard(
                text_data, self.used_ids, longest_utt, self.tokenizer, glove, bert_type
            )
        elif self.d_type == "chalearn" or self.d_type == "firstimpr":
            data_tensor_dict = make_data_tensors_chalearn(
                text_data, self.used_ids, longest_utt, self.tokenizer, glove, bert_type
            )
        elif self.d_type == "cdc":
            data_tensor_dict = make_data_tensors_cdc(
                text_data, self.used_ids, longest_utt, self.tokenizer, glove, bert_type
            )
        elif (
            self.d_type == "mosi"
            or self.d_type == "cmu_mosi"
            or self.d_type == "cmu-mosi"
        ):
            data_tensor_dict = make_data_tensors_mosi(
                text_data, self.used_ids, longest_utt, self.tokenizer, glove, bert_type
            )
        elif self.d_type == "lives":
            data_tensor_dict = make_data_tensors_lives(
                text_data, self.used_ids, longest_utt, self.tokenizer, glove, bert_type
            )
        elif self.d_type == "asist":
            data_tensor_dict = make_data_tensors_asist(
                text_data, self.used_ids, longest_utt, self.tokenizer, glove, bert_type
            )

        return data_tensor_dict


def get_longest_utt(data, tokenizer, use_bert=False):
    """
    Get the length of longest utterance in the dataset
    :param data: a pandas df containing all utterances
    :param tokenizer: a tokenizer
    :param use_bert: whether to use bert/distilbert tokenizer
    :return: length of longest utterance
    """
    all_utts = data["utterance"].tolist()

    if use_bert:
        # tokenize and count all items in dataset
        item_lens = [
            len(tokenizer.tokenize("[CLS] " + str(utt) + " [SEP]")) for utt in all_utts
        ]
    else:
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


def get_lives_speakers(df):
    # get lives speakers by combining the 'speaker' category with the 'sid' category
    df['speaker'] = df['speaker'].astype(str)
    all_speakers = df[['speaker', 'sid']].agg('-'.join, axis=1)

    return set(all_speakers.unique())


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
    for feats_file in tqdm(glob.glob(f"{file_path}/{feature_set}/*_{feature_set}.csv"), desc=f"Loading acoustic feature files for {dataset}"):
        # read each file as a pandas df
        if use_cols is None or (
            len(use_cols) == 1
            and (use_cols[0].lower() == "none" or use_cols.lower() == "all")
        ):
            feats = pd.read_csv(feats_file, sep=";")
            feats.drop(["name", "frameTime"], axis=1, inplace=True)
        else:
            feats = pd.read_csv(feats_file, usecols=use_cols, sep=";")

        # get the id
        feats_file_name = feats_file.split("/")[-1]

        if dataset == "meld":
            dia_id, utt_id = feats_file_name.split("_")[:2]
            id = (dia_id, utt_id)
        elif dataset == "cdc":
            id = feats_file_name.split(f"_")[1]
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


def make_acoustic_dict_with_custom_features(
    file_path, feats_file_name, dataset, use_cols=None
):
    """
    Make the acoustic dict when non-openSMILE features are used
    :param file_path: the path to the file containing all features for this dataset
    :param feats_file_name: the name of the file containing acoustic features
    :param dataset: the name of the dataset
    :param use_cols: a list of columns to select, or None to get all columns
    :return: an acoustic dict and the acoustic lengths (all 1s)
    """
    print(f"Starting (custom) acoustic dict at {datetime.datetime.now()}")

    # get the acoustic features file
    feats_file = f"{file_path}/{feats_file_name}"

    if use_cols is None or (
        len(use_cols) == 1
        and (use_cols[0].lower() == "none" or use_cols.lower() == "all")
    ):
        feats = pd.read_csv(feats_file, sep=",")
    else:
        feats = pd.read_csv(feats_file, usecols=use_cols, sep=",")

    # change name of audio id if using meld
    if dataset == "meld":
        feats["file_name"] = feats["file_name"].apply(
            lambda x: (x.split("_")[0], x.split("_")[1])
        )
    elif dataset == "cdc":
        feats["file_name"] = feats["file_name"].apply(lambda x: x.split("_")[1])
    elif dataset == "firstimpr":
        # custom dataset converted . to _ between fname and clip num
        feats["file_name"] = feats["file_name"].apply(
            lambda x: ".00".join(x.split("_00"))
        )

    # set ID as index
    feats = feats.set_index("file_name")

    # replace all --undefined-- with 0.0
    feats.replace("--undefined--", 0.0, inplace=True)

    # transpose and convert to dict
    acoustic_dict = feats.T.to_dict("list")

    # convert each value to a DataFrame
    acoustic_dict = {
        key: pd.DataFrame(value).T.astype(float) for key, value in acoustic_dict.items()
    }

    # create parallel dict for acoustic lengths
    acoustic_lengths = {key: 1 for key in acoustic_dict.keys()}

    # return
    return acoustic_dict, acoustic_lengths


def make_spectrograms_dict(file_path, dataset):
    """
    Make a dict of spectrograms to include in data
    Uses pre-saved spectrogram CSV files
    :param file_path: the path to the file containing all features for this dataset
    :param dataset: the name of the dataset
    :return: a dict of spectograms, dict of length of each spectrogram
    """
    print(f"Starting spectrogram dict at {datetime.datetime.now()}")

    spec_dict = {}
    spec_lengths = {}

    # get the acoustic features files
    for spec_file in tqdm(glob.glob(f"{file_path}/spec/*.csv"), desc=f"Loading spectrogram files for {dataset}"):
        # read each file as a pandas df
        spec = pd.read_csv(spec_file)

        # get the id
        feats_file_name = spec_file.split("/")[-1]

        if dataset == "meld":
            dia_id, utt_id = feats_file_name.split(".csv")[0].split("_")
            id = (dia_id, utt_id)
        elif dataset == "cdc":
            id = feats_file_name.split(".csv")[0].split("_")[1]
        else:
            id = feats_file_name.split(".csv")[0]

        # save the dataframe to a dict with id as key
        if spec.shape[0] > 0:
            spec_dict[id] = spec
            # do this so we can ensure same order of lengths and feats
            spec_lengths[id] = spec.shape[0]

        # delete the features df bc it takes up lots of space
        del spec

    print(f"Acoustic dict made at {datetime.datetime.now()}")
    print(f"Len of dict: {len(spec_dict.keys())}")
    return spec_dict, spec_lengths


def get_distilbert_tokenizer():
    # instantiate distilbert emb object
    distilbert_emb_mkr = DistilBertEmb()

    # return tokenizer
    return distilbert_emb_mkr.tokenizer


def get_bert_tokenizer():
    # instantiate bert emb onject
    bert_emb_mkr = BertEmb()

    # return
    return bert_emb_mkr.tokenizer
