from cdc.prep_cdc import *
from cmu_mosi.prep_mosi import *
from firstimpr.prep_firstimpr import *
from lives_health.prep_lives import prep_lives_data
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
    avg_acoustic_data=False,
    custom_feats_file=None,
    selected_ids=None,
    num_train_ex=None,
    include_spectrograms=False,
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
        emb_type,
        feats_to_use,
        pred_type,
        data_as_dict,
        avg_acoustic_data,
        custom_feats_file,
        selected_ids=selected_ids,
        num_train_ex=num_train_ex,
        include_spectrograms=include_spectrograms
    )

    # use custom feats set instead of ISXX in save name
    #   if custom feats are used
    if custom_feats_file is not None:
        feature_set = custom_feats_file.split(".")[0]

    if data_as_dict:
        dtype = "dict"
    else:
        dtype = "list"

    if include_spectrograms:
        train_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_spec_train"
        dev_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_spec_dev"
        test_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_spec_test"
        wts_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_spec_clsswts"
    else:
        train_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_train"
        dev_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_dev"
        test_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_test"
        wts_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_clsswts"

    if zip:
        pickle.dump(train_ds, bz2.BZ2File( f"{train_save_name}.bz2", "wb"))
        pickle.dump(dev_ds, bz2.BZ2File(f"{dev_save_name}.bz2", "wb"))
        pickle.dump(test_ds, bz2.BZ2File(f"{test_save_name}.bz2", "wb"))
        pickle.dump(clss_weights, bz2.BZ2File(f"{wts_save_name}.bz2", "wb"))
    else:
        pickle.dump(train_ds,open(f"{train_save_name}.pickle", "wb"))
        pickle.dump(dev_ds, open(f"{dev_save_name}.pickle", "wb"))
        pickle.dump(test_ds, open(f"{test_save_name}.pickle", "wb"))
        pickle.dump(clss_weights, open(f"{wts_save_name}.pickle", "wb"))


def prep_data(
    dataset,
    data_path,
    feature_set,
    transcription_type,
    glove_path,
    emb_type,
    feats_to_use,
    pred_type=None,
    data_as_dict=False,
    avg_acoustic_data=False,
    custom_feats_file=None,
    selected_ids=None,
    num_train_ex=None,
    include_spectrograms=False,
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
            data_path,
            feature_set,
            transcription_type,
            emb_type,
            glove_path,
            feats_to_use,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            num_train_ex=num_train_ex,
            include_spectrograms=include_spectrograms,
        )
    elif dataset == "mosi" or dataset == "cmu_mosi" or dataset == "cmu-mosi":
        train, dev, test, weights = prep_mosi_data(
            data_path,
            feature_set,
            transcription_type,
            emb_type,
            glove_path,
            feats_to_use,
            pred_type,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            num_train_ex=num_train_ex,
            include_spectrograms=include_spectrograms
        )
    elif dataset == "firstimpr" or dataset == "chalearn":
        train, dev, test, weights = prep_firstimpr_data(
            data_path,
            feature_set,
            transcription_type,
            emb_type,
            glove_path,
            feats_to_use,
            pred_type,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            num_train_ex=num_train_ex,
            include_spectrograms=include_spectrograms
        )
    elif dataset == "meld":
        train, dev, test, weights = prep_meld_data(
            data_path,
            feature_set,
            transcription_type,
            emb_type,
            glove_path,
            feats_to_use,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            num_train_ex=num_train_ex,
            include_spectrograms=include_spectrograms
        )
    elif dataset == "mustard":
        train, dev, test, weights = prep_mustard_data(
            data_path,
            feature_set,
            transcription_type,
            emb_type,
            glove_path,
            feats_to_use,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            num_train_ex=num_train_ex,
            include_spectrograms=include_spectrograms
        )
    elif dataset == "ravdess":
        train, dev, test, weights = prep_ravdess_data(
            data_path,
            feature_set,
            emb_type,
            glove_path,
            feats_to_use,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            selected_ids=selected_ids,
            num_train_ex=num_train_ex,
            include_spectrograms=include_spectrograms
        )
    elif dataset == "lives":
        train, dev, test, weights = prep_lives_data(
            data_path,
            feature_set,
            transcription_type,
            emb_type,
            glove_path,
            feats_to_use,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            num_train_ex=num_train_ex,
            include_spectrograms=include_spectrograms
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
    firstimpr_path = f"{base_path}/FirstImpr"
    meld_path = f"{base_path}/MELD_formatted"
    mustard_path = f"{base_path}/MUStARD"
    ravdess_path = f"{base_path}/RAVDESS_Speech"
    lives_path = "../../lives_test/done"

    save_path = "../../datasets/pickled_data"

    glove_path = "../../datasets/glove/glove.subset.300d.txt"

    feature_set = "IS13"

    # get the ids from a pickle file containing ids in order
    selected_ids_dict = pickle.load(
        open("../../datasets/pickled_data/ravdess_ordered_ids.pickle", "rb")
    )
    selected_ids = []
    selected_ids.extend(selected_ids_dict["train"])
    selected_ids.extend(selected_ids_dict["test"])
    selected_ids.extend(selected_ids_dict["dev"])

    transcription_type = "gold"
    emb_type = "glove"
    # emb_type = "distilbert"
    # emb_type = "bert"
    dict_data = True
    avg_feats = True
    with_spec = True

    # datasets = ["cdc", "mosi", "firstimpr", "meld", "ravdess"]
    # datasets = ["firstimpr", "meld", "ravdess"]
    # datasets = ["mustard"]
    datasets = ["lives"]

    # custom_feats_file = "combined_features_small.txt"
    custom_feats_file = None

    # set number of training examples
    # num_train = 500
    num_train = None

    for dataset in datasets:
        if dataset == "mosi":
            save_partitioned_data(
                dataset,
                save_path,
                mosi_path,
                feature_set,
                transcription_type,
                glove_path,
                pred_type="classification",
                emb_type=emb_type,
                data_as_dict=dict_data,
                avg_acoustic_data=avg_feats,
                custom_feats_file=custom_feats_file,
                num_train_ex=num_train,
                include_spectrograms=with_spec,
            )
        elif dataset == "firstimpr":
            save_partitioned_data(
                dataset,
                save_path,
                firstimpr_path,
                feature_set,
                transcription_type,
                glove_path,
                pred_type="max_class",
                emb_type=emb_type,
                data_as_dict=dict_data,
                avg_acoustic_data=avg_feats,
                custom_feats_file=custom_feats_file,
                num_train_ex=num_train,
                include_spectrograms=with_spec,
            )
        elif dataset == "cdc":
            save_partitioned_data(
                dataset,
                save_path,
                cdc_path,
                feature_set,
                transcription_type,
                glove_path,
                emb_type=emb_type,
                data_as_dict=dict_data,
                avg_acoustic_data=avg_feats,
                custom_feats_file=custom_feats_file,
                num_train_ex=num_train,
                include_spectrograms=with_spec,
            )
        elif dataset == "meld":
            save_partitioned_data(
                dataset,
                save_path,
                meld_path,
                feature_set,
                transcription_type,
                glove_path,
                emb_type=emb_type,
                data_as_dict=dict_data,
                avg_acoustic_data=avg_feats,
                custom_feats_file=custom_feats_file,
                num_train_ex=num_train,
                include_spectrograms=with_spec,
            )
        elif dataset == "mustard":
            save_partitioned_data(
                dataset,
                save_path,
                mustard_path,
                feature_set,
                transcription_type,
                glove_path,
                emb_type=emb_type,
                data_as_dict=dict_data,
                avg_acoustic_data=avg_feats,
                custom_feats_file=custom_feats_file,
                num_train_ex=num_train,
                include_spectrograms=with_spec,
            )
        elif dataset == "ravdess":
            save_partitioned_data(
                dataset,
                save_path,
                ravdess_path,
                feature_set,
                transcription_type,
                glove_path,
                emb_type=emb_type,
                data_as_dict=dict_data,
                avg_acoustic_data=avg_feats,
                custom_feats_file=custom_feats_file,
                selected_ids=selected_ids,
                num_train_ex=num_train,
                include_spectrograms=with_spec,
            )
        elif dataset == "lives":
            save_partitioned_data(
                dataset,
                save_path,
                lives_path,
                feature_set,
                transcription_type,
                glove_path,
                emb_type=emb_type,
                data_as_dict=dict_data,
                avg_acoustic_data=avg_feats,
                custom_feats_file=custom_feats_file,
                selected_ids=selected_ids,
                num_train_ex=num_train,
                include_spectrograms=with_spec,
            )
