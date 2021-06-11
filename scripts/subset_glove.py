# use this to create a subset of glove for the relevant data
# to save time during dataset creation, model training and testing

import glove.glove_subsetting as glove
import pandas as pd
import numpy as np
import argparse
from torchtext.data import get_tokenizer

from utils.data_prep_helpers import clean_up_word

parser = argparse.ArgumentParser()
parser.add_argument(
    "--glove_path",
    help="Path to full GloVe file",
    default="../../glove.42B.300d.txt",
    nargs="?",
)

parser.add_argument(
    "--save_path_and_name",
    help="Path and name of file to save",
    default="../../glove.subset.300d.txt",
    nargs="?",
)

parser.add_argument(
    "--vector_length",
    help="The length of glove vectors in file of interest",
    default=300,
    nargs="?",
)

parser.add_argument(
    "--utterance_files",
    help="The files containing a column 'utterance' with each utterance",
    nargs="+",
)

args = parser.parse_args()

if __name__ == "__main__":
    # get list of all files to load
    all_files = args.utterance_files

    # get tokenizer
    tokenizer = get_tokenizer("basic_english")

    # prep holder for all utterances
    vocab = set()

    # start by adding in the words from ravdess
    ravdess_utts = ["kids are talking by the door", "dogs are sitting by the door"]

    for utt in ravdess_utts:
        wds = tokenizer(utt)

        for wd in wds:
            vocab.add(wd)

    # iterate through files to get utterances
    for f in all_files:
        try:
            f_pd = pd.read_csv(f, sep="\t")
            f_pd.columns = f_pd.columns.str.lower()

            utts = f_pd.utterance
            utts = utts.replace(np.nan, "", regex=True)
        except AttributeError:
            f_pd = pd.read_csv(f, sep=",")
            f_pd.columns = f_pd.columns.str.lower()

            utts = f_pd.utterance
            utts = utts.replace(np.nan, "", regex=True)

        for utt in utts:
            utt = clean_up_word(utt)
            wds = tokenizer(utt)

            for wd in wds:
                vocab.add(wd)

    # subset glove
    subset = glove.subset_glove(
        args.glove_path, vocab, args.vector_length, add_unk=True
    )

    # save the subset
    glove.save_subset(subset, args.save_path_and_name)
