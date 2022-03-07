from cdc.prep_cdc import *
from cmu_mosi.prep_mosi import *
from utils.data_saving_and_loading_helpers import *

import pickle
import bz2
import os


def save_data_components(
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
    custom_feats_file=None,
    selected_ids=None,
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
        True,  # data must be dict to use this saving format
        False,  # acoustic data must not be averaged
        custom_feats_file,
        selected_ids=selected_ids,
        num_train_ex=None,
        include_spectrograms=True
    )
    # save class weights
    if zip:
        pickle.dump(clss_weights, bz2.BZ2File(f"{save_path}/{dataset}_clsswts.bz2", "wb"))
    else:
        pickle.dump(clss_weights, open(f"{save_path}/{dataset}_clsswts.pickle", "wb"))


    # todo: implement if we need custom fields per dataset
    # # get all parts of data to save
    # all_fields = dev_ds[0].keys()

    all_data = [('train', train_ds),
                ('dev', dev_ds),
                ('test', test_ds)]

    for partition_tuple in all_data:
        # get name of partition
        partition_name = partition_tuple[0]
        partition = partition_tuple[1]
        # get spec data + audio_ids
        spec_data = get_specific_fields(partition, "spec")
        spec_save_name = f"{save_path}/{dataset}_spec_{partition_name}"

        # get acoustic data + audio_ids
        acoustic_data = get_specific_fields(partition, "acoustic")
        # use custom feats set instead of ISXX in save name
        #   if custom feats are used
        if custom_feats_file is not None:
            feature_set = custom_feats_file.split(".")[0]
        acoustic_save_name = f"{save_path}/{dataset}_{feature_set}_{partition_name}"

        # get utt data + audio_ids
        utt_data = get_specific_fields(partition, "utt")
        utt_save_name = f"{save_path}/{dataset}_{emb_type}_{partition_name}"

        # get ys data + audio_ids
        ys_data = get_specific_fields(partition, "ys")
        ys_save_name = f"{save_path}/{dataset}_ys_{partition_name}"

        # save
        if zip:
            pickle.dump(spec_data, bz2.BZ2File(f"{spec_save_name}.bz2", "wb"))
            pickle.dump(acoustic_data, bz2.BZ2File(f"{acoustic_save_name}.bz2", "wb"))
            pickle.dump(utt_data, bz2.BZ2File(f"{utt_save_name}.bz2", "wb"))
            pickle.dump(ys_data, bz2.BZ2File(f"{ys_save_name}.bz2", "wb"))
        else:
            pickle.dump(spec_data,open(f"{spec_save_name}.pickle", "wb"))
            pickle.dump(acoustic_data, open(f"{acoustic_save_name}.pickle", "wb"))
            pickle.dump(utt_data, open(f"{utt_save_name}.pickle", "wb"))
            pickle.dump(ys_data, open(f"{ys_save_name}.pickle", "wb"))


def get_specific_fields(data, field_type, fields=None):
    """
    Partition the data based on a set of keys
    :param data: The dataset
    :param field_type: 'spec', 'acoustic', 'utt', 'ys', or 'other'
    :param fields: if specific fields are given, use this instead of
        field type to get portions of data
    :return: The subset of data with these fields
    """
    sub_data = []
    if fields is not None:
        for item in data:
            sub_data.append({key: value for key, value in item.items() if key in fields})
    else:
        if field_type.lower() == "spec":
            keys = ["x_spec", "spec_length", "audio_id"]
        elif field_type.lower() == "acoustic":
            keys = ["x_acoustic", "acoustic_length", "audio_id"]
        elif field_type.lower() == "utt":
            keys = ["x_utt", "utt_length", "audio_id"]
        elif field_type.lower() == "ys":
            keys = ["ys", "audio_id"]
        else:
            exit("Field type not listed, and no specific fields given")

        for item in data:
            sub_data.append({key: value for key, value in item.items() if key in keys})

    return sub_data


if __name__ == "__main__":
    base_path = "../../datasets/multimodal_datasets"
    cdc_path = f"{base_path}/Columbia_deception_corpus"
    mosi_path = f"{base_path}/CMU_MOSI"
    firstimpr_path = f"{base_path}/FirstImpr"
    meld_path = f"{base_path}/MELD_formatted"
    mustard_path = f"{base_path}/MUStARD"
    ravdess_path = f"{base_path}/RAVDESS_Speech"
    lives_path = "../../lives_test/done"

    save_path = "../../datasets/pickled_data/field_separated_data"

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

    # datasets = ["cdc", "mosi", "firstimpr", "meld", "ravdess"]
    # datasets = ["mosi"]
    datasets = ["cdc", "firstimpr", "meld", "ravdess"]
    # datasets = ["ravdess"]
    # datasets = ["lives"]

    # custom_feats_file = "combined_features_small.txt"
    custom_feats_file = None

    for dataset in datasets:
        if dataset == "mosi":
            save_data_components(
                dataset,
                save_path,
                mosi_path,
                feature_set,
                transcription_type,
                glove_path,
                pred_type="classification",
                emb_type=emb_type,
                custom_feats_file=custom_feats_file,
            )
        elif dataset == "firstimpr":
            save_data_components(
                dataset,
                save_path,
                firstimpr_path,
                feature_set,
                transcription_type,
                glove_path,
                pred_type="max_class",
                emb_type=emb_type,
                custom_feats_file=custom_feats_file,
            )
        elif dataset == "cdc":
            save_data_components(
                dataset,
                save_path,
                cdc_path,
                feature_set,
                transcription_type,
                glove_path,
                emb_type=emb_type,
                custom_feats_file=custom_feats_file,
            )
        elif dataset == "meld":
            save_data_components(
                dataset,
                save_path,
                meld_path,
                feature_set,
                transcription_type,
                glove_path,
                emb_type=emb_type,
                custom_feats_file=custom_feats_file,
            )
        elif dataset == "mustard":
            save_data_components(
                dataset,
                save_path,
                mustard_path,
                feature_set,
                transcription_type,
                glove_path,
                emb_type=emb_type,
                custom_feats_file=custom_feats_file,
            )
        elif dataset == "ravdess":
            save_data_components(
                dataset,
                save_path,
                ravdess_path,
                feature_set,
                transcription_type,
                glove_path,
                emb_type=emb_type,
                custom_feats_file=custom_feats_file,
                selected_ids=selected_ids,
            )
        # todo: lives requires custom fields
        # elif dataset == "lives":
        #     save_data_components(
        #         dataset,
        #         save_path,
        #         lives_path,
        #         feature_set,
        #         transcription_type,
        #         glove_path,
        #         emb_type=emb_type,
        #         custom_feats_file=custom_feats_file,
        #         selected_ids=selected_ids,
        #     )
