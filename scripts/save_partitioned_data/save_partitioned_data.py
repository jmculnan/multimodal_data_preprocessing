from cdc.prep_cdc import *
from cmu_mosi.prep_mosi import *
from firstimpr.prep_firstimpr import *
from meld.prep_meld import *
from mustard.prep_mustard import *
from ravdess.prep_ravdess import *

import pickle
import bz2
import os

from torch.utils.data import Dataset


def save_partitioned_data(dataset, save_path, data_path, feature_set,
                          transcription_type, glove_path, feats_to_use=None,
                          pred_type=None, zip=False):
    """
    Save partitioned data in pickled format
    :param dataset: the string name of dataset to use
    :param save_path: path where you want to save pickled data
    :param data_path: path to the data
    :param feature_set: IS09-13
    :param transcription_type: Gold, Google, Kaldi, Sphinx
    :param glove_path: path to glove file
    :param feats_to_use: list of features, if needed
    :param pred_type: type of predictions, for mosi and firstimpr
    :parak zip: whether to save as a bz2 compressed file
    :return:
    """
    dataset = dataset.lower()

    # make sure the full save path exists; if not, create it
    os.system(f'if [ ! -d "{save_path}" ]; then mkdir -p {save_path}; fi')

    train_ds, dev_ds, test_ds, clss_weights = prep_data(dataset, data_path, feature_set, transcription_type,
                                                        glove_path, feats_to_use, pred_type)

    if zip:
        pickle.dump(train_ds, bz2.BZ2File(f"{save_path}/{dataset}_{feature_set}_train.bz2", "wb"))
        pickle.dump(dev_ds, bz2.BZ2File(f"{save_path}/{dataset}_{feature_set}_dev.bz2", "wb"))
        pickle.dump(test_ds, bz2.BZ2File(f"{save_path}/{dataset}_{feature_set}_test.bz2", "wb"))
        pickle.dump(clss_weights, bz2.BZ2File(f"{save_path}/{dataset}_{feature_set}_clsswts.bz2", "wb"))
    else:
        pickle.dump(train_ds, open(f"{save_path}/{dataset}_{feature_set}_train.pickle", "wb"))
        pickle.dump(dev_ds, open(f"{save_path}/{dataset}_{feature_set}_dev.pickle", "wb"))
        pickle.dump(test_ds, open(f"{save_path}/{dataset}_{feature_set}_test.pickle", "wb"))
        pickle.dump(clss_weights, open(f"{save_path}/{dataset}_{feature_set}_clsswts.pickle", "wb"))


def prep_data(dataset, data_path, feature_set,
              transcription_type, glove_path, feats_to_use,
              pred_type=None):
    """
    Prepare data for a given dataset
    :param dataset: string name of dataset
    :return:
    """
    dataset = dataset.lower()

    if dataset == "cdc":
        train, dev, test, weights = prep_cdc_data(data_path, feature_set, transcription_type,
                                                  glove_path, feats_to_use)
    elif dataset == "mosi" or dataset == "cmu_mosi" or dataset == "cmu-mosi":
        train, dev, test, weights = prep_mosi_data(data_path, feature_set, transcription_type,
                                                   glove_path, feats_to_use, pred_type)
    elif dataset == "firstimpr" or dataset == "chalearn":
        train, dev, test, weights = prep_firstimpr_data(data_path, feature_set, transcription_type,
                                                        glove_path, feats_to_use, pred_type)
    elif dataset == "meld":
        train, dev, test, weights = prep_meld_data(data_path, feature_set, transcription_type,
                                                   glove_path, feats_to_use)
    elif dataset == "mustard":
        train, dev, test, weights = prep_mustard_data(data_path, feature_set, transcription_type,
                                                      glove_path, feats_to_use)
    elif dataset == "ravdess":
        train, dev, test, weights = prep_ravdess_data(data_path, feature_set, glove_path, feats_to_use)

    return train, dev, test, weights


class DatumListDataset(Dataset):
    """
    A dataset to hold a list of datums
    """

    def __init__(self, data_list, data_type="meld_emotion", class_weights=None):
        self.data_list = data_list
        self.data_type = data_type
        # todo: add task number

        self.class_weights = class_weights

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        """
        item (int) : the index to a data point
        """
        return self.data_list[item]

    def targets(self):
        if (
            self.data_type == "meld_emotion"
            or self.data_type == "mustard"
            or self.data_type == "ravdess_emotion"
        ):
            for datum in self.data_list:
                yield datum[4]
        elif (
            self.data_type == "meld_sentiment" or self.data_type == "ravdess_intensity"
        ):
            for datum in self.data_list:
                yield datum[5]


if __name__ == "__main__":
    base_path = "../../datasets/multimodal_datasets"
    cdc_path = f"{base_path}/Columbia_deception_corpus"
    mosi_path = f"{base_path}/CMU_MOSI"
    firstimpr_path = f"{base_path}/Chalearn"
    meld_path = f"{base_path}/MELD_formatted"
    mustard_path = f"{base_path}/MUStARD"
    ravdess_path = f"{base_path}/RAVDESS_Speech"

    save_path = "../../datasets/pickled_data"

    glove_path = "../../glove.short.300d.punct.txt"

    feature_set = "IS13"

    transcription_type = "Gold"

    # save_partitioned_data("cdc", save_path, cdc_path, feature_set, transcription_type,
    #                       glove_path)

    # save_partitioned_data("mosi", save_path, mosi_path, feature_set, transcription_type,
    #                       glove_path, pred_type="classification")
    #
    # save_partitioned_data("firstimpr", save_path, firstimpr_path, feature_set, transcription_type,
    #                       glove_path, pred_type="max_class")
    #
    # save_partitioned_data("meld", save_path, meld_path, feature_set, transcription_type,
    #                       glove_path)
    #
    # save_partitioned_data("mustard", save_path, mustard_path, feature_set, transcription_type,
    #                       glove_path)

    save_partitioned_data("ravdess", save_path, ravdess_path, feature_set, transcription_type,
                          glove_path)
