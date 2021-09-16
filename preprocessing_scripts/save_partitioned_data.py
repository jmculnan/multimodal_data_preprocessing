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


def save_partitioned_data(
    dataset,
    save_path,
    data_path,
    feature_set,
    transcription_type,
    glove_path,
    emb_type,
    feats_to_use=None,
    pred_type=None,
    zip=False,
    data_as_dict=False,
    avg_acoustic_data=False
):
    """
    Save partitioned data in pickled format
    :param dataset: the string name of dataset to use
    :param save_path: path where you want to save pickled data
    :param data_path: path to the data
    :param feature_set: IS09-13
    :param transcription_type: Gold, Google, Kaldi, Sphinx
    :param glove_path: path to glove file
    :param emb_type: whether to use glove or distilbert
    :param feats_to_use: list of features, if needed
    :param pred_type: type of predictions, for mosi and firstimpr
    :param zip: whether to save as a bz2 compressed file
    :param data_as_dict: whether each datapoint saves as a dict
    :return:
    """
    dataset = dataset.lower()

    # make sure the full save path exists; if not, create it
    os.system(f'if [ ! -d "{save_path}" ]; then mkdir -p {save_path}; fi')

    train_ds, dev_ds, test_ds, clss_weights = prep_data(
        dataset,
        data_path,
        feature_set,
        transcription_type,
        glove_path,
        feats_to_use,
        pred_type,
        data_as_dict,
        avg_acoustic_data
    )

    if data_as_dict:
        dtype = "dict"
    else:
        dtype = "list"

    if zip:
        pickle.dump(
            train_ds,
            bz2.BZ2File(f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_train.bz2", "wb"),
        )
        pickle.dump(
            dev_ds, bz2.BZ2File(f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_dev.bz2", "wb")
        )
        pickle.dump(
            test_ds, bz2.BZ2File(f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_test.bz2", "wb")
        )
        pickle.dump(
            clss_weights,
            bz2.BZ2File(f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_clsswts.bz2", "wb"),
        )
    else:
        pickle.dump(
            train_ds, open(f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_train.pickle", "wb")
        )
        pickle.dump(
            dev_ds, open(f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_dev.pickle", "wb")
        )
        pickle.dump(
            test_ds, open(f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_test.pickle", "wb")
        )
        pickle.dump(
            clss_weights,
            open(f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_clsswts.pickle", "wb"),
        )


def prep_data(
    dataset,
    data_path,
    feature_set,
    transcription_type,
    glove_path,
    feats_to_use,
    pred_type=None,
    data_as_dict=False,
    avg_acoustic_data=False
):
    """
    Prepare data for a given dataset
    :param dataset: string name of dataset
    :return:
    """
    dataset = dataset.lower()

    print("-------------------------------------------")
    print(f"Starting dataset prep for {dataset}")
    print("-------------------------------------------")

    if dataset == "cdc":
        train, dev, test, weights = prep_cdc_data(
            data_path, feature_set, transcription_type, "distilbert", glove_path, feats_to_use,
            as_dict=data_as_dict, avg_acoustic_data=avg_acoustic_data
        )
    elif dataset == "mosi" or dataset == "cmu_mosi" or dataset == "cmu-mosi":
        train, dev, test, weights = prep_mosi_data(
            data_path,
            feature_set,
            transcription_type,
            "distilbert",
            glove_path,
            feats_to_use,
            pred_type,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data
        )
    elif dataset == "firstimpr" or dataset == "chalearn":
        train, dev, test, weights = prep_firstimpr_data(
            data_path,
            feature_set,
            transcription_type,
            "distilbert",
            glove_path,
            feats_to_use,
            pred_type,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data
        )
    elif dataset == "meld":
        train, dev, test, weights = prep_meld_data(
            data_path, feature_set, transcription_type, "distilbert", glove_path, feats_to_use,
            as_dict=data_as_dict, avg_acoustic_data=avg_acoustic_data
        )
    elif dataset == "mustard":
        train, dev, test, weights = prep_mustard_data(
            data_path, feature_set, transcription_type, "distilbert", glove_path, feats_to_use,
            as_dict=data_as_dict, avg_acoustic_data=avg_acoustic_data
        )
    elif dataset == "ravdess":
        train, dev, test, weights = prep_ravdess_data(
            data_path, feature_set, "distilbert", glove_path, feats_to_use,
            as_dict=data_as_dict, avg_acoustic_data=avg_acoustic_data
        )

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

    glove_path = "../../datasets/glove/glove.subset.300d.txt"

    feature_set = "IS13"

    transcription_type = "gold"

    # save_partitioned_data(
    #     "cdc", save_path, cdc_path, feature_set, transcription_type, glove_path, emb_type="distilbert",
    #     data_as_dict=True
    # )
    #
    # save_partitioned_data(
    #     "mosi",
    #     save_path,
    #     mosi_path,
    #     feature_set,
    #     transcription_type,
    #     glove_path,
    #     pred_type="classification",
    #     emb_type="distilbert",
    #     data_as_dict=True
    # )
    #
    # save_partitioned_data(
    #     "firstimpr",
    #     save_path,
    #     firstimpr_path,
    #     feature_set,
    #     transcription_type,
    #     glove_path,
    #     pred_type="max_class",
    #     emb_type="distilbert",
    #     data_as_dict=True
    # )
    #
    # save_partitioned_data(
    #     "meld", save_path, meld_path, feature_set, transcription_type, glove_path, emb_type="distilbert",
    #     data_as_dict=True
    # )
    #
    # save_partitioned_data(
    #     "mustard", save_path, mustard_path, feature_set, transcription_type, glove_path, emb_type="distilbert",
    #     data_as_dict=True
    # )

    save_partitioned_data(
        "ravdess", save_path, ravdess_path, feature_set, transcription_type, glove_path, emb_type="distilbert",
        data_as_dict=True, avg_acoustic_data=False
    )
