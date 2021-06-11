# get the subset of GloVe that relates to the vocabulary present in the texts
# this should allow for faster usage later

import os
import pickle

import pandas as pd
import numpy as np
from utils.data_prep_helpers import clean_up_word, make_glove_dict, Glove

from torchtext.data import get_tokenizer


def get_all_vocab(data_dir):
    """
    Get all the words in the vocabulary from a given directory
    :param data_dir:
    :return:
    """
    # save to set
    all_vocab = set()
    #
    for f in os.listdir(data_dir):
        if f.endswith("IS09_avgd.csv"):
            wds = pd.read_csv(data_dir + "/" + f, usecols=["word"])
            wds = wds["word"].tolist()
            for item in wds:
                item = clean_up_word(item)
                all_vocab.add(item)
    return all_vocab


def get_all_vocab_from_transcribed_df(dataframe):
    """
    Get all the words in the vocabulary from a transcribed file
    dataframe: a pandas df containing a column 'utterance'
    """
    all_wds = []
    all_vocab = set()
    # get tokenizer
    tokenizer = get_tokenizer("basic_english")
    # put all utterances in a list
    utts = dataframe["utterance"].tolist()
    for utt in utts:
        try:
            utt = clean_up_word(utt)
            wds = tokenizer(utt)
        except AttributeError:
            print(utt)
        for wd in wds:
            all_vocab.add(wd)
            all_wds.append(wd)

    return all_vocab


def compare_vocab_with_existing_data(vocab_set, existing_set):
    """
    Compare a vocabulary set with existing data
    Used for appending new embeddings to file
    existing_set : a set of existing data
    """
    for wd in existing_set:
        if wd in vocab_set:
            vocab_set.remove(wd)

    return vocab_set


def read_glove(glove_path):
    """
    Read the GloVe file
    pandas read_csv cannot handle this file
    because of ' and " as items
    """
    word2glove = {}
    with open(glove_path, "r") as glove:
        for line in glove:
            vals = line.split()
            word2glove[vals[0]] = np.array(float(item) for item in vals[1:])

    return word2glove


def subset_glove(glove_path, vocab_set, vec_len=100, add_unk=False):
    with open(glove_path, "r") as glove:
        subset = []
        num_items = 0
        if add_unk:
            unk_vec = np.zeros(vec_len)
        for line in glove:
            vals = line.split()
            if vals[0] in vocab_set:
                num_items += 1
                subset.append(vals)
                if add_unk:
                    unk_vec = unk_vec + np.array([float(item) for item in vals[1:]])
    if add_unk:
        unk_vec = unk_vec / num_items
        unk_vec = ["<UNK>"] + unk_vec.tolist()
        unk_vec = [str(item) for item in unk_vec]
        subset.append(unk_vec)

    return subset


def save_subset(subset, save_path):
    with open(save_path, "w") as gfile:
        for item in subset:
            gfile.write(" ".join(item))
            gfile.write("\n")


def append_subset(subset, save_path):
    with open(save_path, "a") as gfile:
        for item in subset:
            gfile.write(" ".join(item))
            gfile.write("\n")


def pickle_glove(glove_object, save_path):
    """
    Save a pickle of the glove object
    :param glove_object: instance of class Glove
    :param save_path: path to save object
    :return:
    """
    pickle.dump(glove_object, open(f"{save_path}", "wb"))
