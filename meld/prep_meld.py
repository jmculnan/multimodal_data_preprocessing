import argparse
import pickle

from prep_data import *
from utils.data_prep_helpers import Glove, make_glove_dict
from combine_xs_and_ys_by_dataset import combine_xs_and_ys_meld
from make_data_tensors_by_dataset import make_data_tensors_meld


def prep_meld_data(
        data_path="../../datasets/multimodal_datasets/meld_formatted",
        feature_set="IS13",
        transcription_type="gold",
        glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
        features_to_use=None
):
    # load glove
    glove_dict = make_glove_dict(glove_filepath)
    glove = Glove(glove_dict)

    # holder for name of file containing utterance info
    if transcription_type.lower() == "gold":
        utts_name = "sent_emo.csv"
    else:
        utts_name = f"meld_{transcription_type.lower()}.tsv"

    # create instance of StandardPrep class
    meld_prep = StandardPrep(
        data_type="meld",
        data_path=data_path,
        feature_set=feature_set,
        utterance_fname=utts_name,
        glove=glove,
        transcription_type=transcription_type,
        use_cols=features_to_use
    )

    train_data = meld_prep.train_prep.combine_xs_and_ys()
    dev_data = meld_prep.dev_prep.combine_xs_and_ys()
    test_data = meld_prep.test_prep.combine_xs_and_ys()

    class_weights = meld_prep.train_prep.class_weights

    return train_data, dev_data, test_data, class_weights


if __name__ == "__main__":
    train, dev, test, weights = prep_meld_data()

    print(weights)
    print(type(train))