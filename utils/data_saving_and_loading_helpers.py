from asist.prep_asist import prep_asist_data
from cdc.prep_cdc import *
from cmu_mosi.prep_mosi import *
from firstimpr.prep_firstimpr import *
from lives_health.prep_lives import prep_lives_data
from meld.prep_meld import *
from mustard.prep_mustard import *
from ravdess.prep_ravdess import *

from torch.utils.data import Dataset

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
    elif dataset == "asist":
        train, dev, test, weights = prep_asist_data(
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
