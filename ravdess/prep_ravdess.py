# prepare RAVDESS data for input into the model

import os

import torch
from torch import nn
from torchtext.data import get_tokenizer

from utils.audio_extraction import ExtractAudio
import pandas as pd

from prep_data import *
from utils.data_prep_helpers import (
    get_class_weights,
    get_gender_avgs,
    create_data_folds_list,
    Glove,
    make_glove_dict,
)


def prep_ravdess_data(
    data_path="../../datasets/multimodal_datasets/RAVDESS_Speech",
    feature_set="IS13",
    glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
    features_to_use=None,
):
    # load glove
    glove_dict = make_glove_dict(glove_filepath)
    glove = Glove(glove_dict)

    # create instance of StandardPrep class
    ravdess_prep = RavdessPrep(
        ravdess_path=data_path,
        glove=glove,
        train_prop=0.6,
        test_prop=0.2,
        f_end=f"{feature_set}.csv",
        use_cols=features_to_use,
    )

    # get train, dev, test
    train_data = ravdess_prep.normalize_acoustic_data_in_tensors(
        ravdess_prep.train_data
    )
    dev_data = ravdess_prep.normalize_acoustic_data_in_tensors(ravdess_prep.dev_data)
    test_data = ravdess_prep.normalize_acoustic_data_in_tensors(ravdess_prep.test_data)

    # get class weights
    # todo: allow to get emotion or intensity or both
    class_weights = ravdess_prep.emotion_weights

    return train_data, dev_data, test_data, class_weights


class RavdessPrep:
    """
    A class to prepare ravdess data
    """

    def __init__(
        self,
        ravdess_path,
        glove,
        train_prop=0.6,
        test_prop=0.2,
        f_end="IS10.csv",
        use_cols=None,
        add_avging=True,
        avgd=False,
    ):
        # path to dataset--all within acoustic files for ravdess
        self.path = ravdess_path

        # get tokenizer
        self.tokenizer = get_tokenizer("basic_english")

        # get data tensors
        self.all_data = make_ravdess_data_tensors(
            self.path, glove, f_end, use_cols, add_avging=add_avging, avgd=avgd
        )

        (self.train_data, self.dev_data, self.test_data,) = create_data_folds_list(
            self.all_data, train_prop, test_prop
        )

        # pull out ys from train to get class weights
        self.train_y_emotion = torch.tensor([item[4] for item in self.train_data])
        self.train_y_intensity = torch.tensor([item[5] for item in self.train_data])

        # set the sarcasm weights
        self.emotion_weights = get_class_weights(self.train_y_emotion)
        self.intensity_weights = get_class_weights(self.train_y_intensity)

        # pull out acoustic data and gender data from train for normalization
        self.train_acoustic = torch.tensor(
            [item[0].tolist() for item in self.train_data]
        )
        self.train_genders = [item[3] for item in self.train_data]

        # acoustic feature normalization based on train
        self.acoustic_means = self.train_acoustic.mean(dim=0, keepdim=False)
        self.acoustic_stdev = self.train_acoustic.std(dim=0, keepdim=False)
        # self.male_acoustic_means, self.male_deviations = get_gender_avgs(
        #     self.train_acoustic, self.train_genders, gender=2
        # )
        # self.female_acoustic_means, self.female_deviations = get_gender_avgs(
        #     self.train_acoustic, self.train_genders, gender=1
        # )

    def normalize_acoustic_data_in_tensors(self, tensor_to_update):
        """
        Updated data tensors to be normalized by means
        :return:
        """
        updated = tensor_to_update.copy().detach()
        for i, item in enumerate(updated):
            updated[i][0] = transform_acoustic_item(
                updated[i][0], self.acoustic_means, self.acoustic_stdev
            )

        return updated


def make_ravdess_data_tensors(
    acoustic_path, glove, f_end="_IS10.csv", use_cols=None, add_avging=True, avgd=False,
):
    """
    makes data tensors for use in RAVDESS objects
    f_end: end of acoustic file names
    use_cols: if set, should be a list [] of column names to include
    n_to_skip : the number of columns at the start to ignore (e.g. name, time)
    # fixme: acoustic padding needed for this to work
    """
    # holder for the data
    acoustic_holder = []
    acoustic_lengths = []
    emotions = []
    intensities = []
    utterances = []
    repetitions = []
    speakers = []
    genders = []

    # holder for all data
    data = []

    utt_1 = glove.index(["kids", "are", "talking", "by", "the", "door"])
    utt_2 = glove.index(["dogs", "are", "sitting", "by", "the", "door"])

    # will have to do two for loops
    # one to get the longest acoustic df
    # the other to organize data tensors

    # find acoustic features files
    for f in os.listdir(acoustic_path):
        if f.endswith(f_end):
            # set the separator
            separator = ";"

            # read in the file as a dataframe
            if use_cols is not None:
                feats = pd.read_csv(
                    acoustic_path + "/" + f, usecols=use_cols, sep=separator
                )
            else:
                feats = pd.read_csv(acoustic_path + "/" + f, sep=separator)
                if not avgd:
                    feats.drop(["name", "frameTime"], axis=1, inplace=True)

            # get the labels
            all_labels = f.split("_")[0]
            labels_list = all_labels.split("-")

            emotion = int(labels_list[2]) - 1  # to make it zero-based
            intensity = int(labels_list[3]) - 1  # to make it zero based
            utterance = int(labels_list[4])
            repetition = int(labels_list[5])
            speaker = int(labels_list[6])
            if speaker % 2 == 0:
                gender = 1
            else:
                gender = 2

            if utterance % 2 == 0:
                utt = utt_2
            else:
                utt = utt_1

            # save the dataframe to a dict with (dialogue, utt) as key
            if feats.shape[0] > 0:
                # order of items: acoustic, utt, spkr, gender, emotion
                #   intensity, repetition #, utt_length, acoustic_length
                if add_avging:
                    acoustic_holder.append(
                        torch.mean(torch.tensor(feats.values.tolist()), dim=0)
                    )
                else:
                    acoustic_holder.append(torch.tensor(feats.values.tolist()))
                utterances.append(utt)
                speakers.append(speaker)
                genders.append(gender)
                emotions.append(emotion)
                intensities.append(intensity)
                repetitions.append(repetition)
                acoustic_lengths.append(feats.shape[0])

    # convert data to torch tensors
    utterances = torch.tensor(utterances)
    speakers = torch.tensor(utterances)
    genders = torch.tensor(genders)
    emotions = torch.tensor(emotions)
    intensities = torch.tensor(intensities)
    repetitions = torch.tensor(repetitions)
    acoustic_lengths = torch.tensor(acoustic_lengths)

    if not add_avging:
        acoustic_holder = nn.utils.rnn.pad_sequence(
            acoustic_holder, batch_first=True, padding_value=0
        )

    for i in range(len(acoustic_holder)):
        data.append(
            (
                acoustic_holder[i],
                utterances[i],
                speakers[i],
                genders[i],
                emotions[i],
                intensities[i],
                repetitions[i],
                6,
                acoustic_lengths[i],
            )
        )

    return data


def preprocess_ravdess_data(
    base_path, acoustic_save_dir, smile_path, acoustic_feature_set="IS10"
):
    """
    Preprocess the ravdess data by extracting acoustic features from wav files
    base_path : the path to the base RAVDESS directory
    acoustic_save_dir : the directory in which to save acoustic feature files
    smile_path : the path to OpenSMILE
    acoustic_feature_set : the feature set to use with ExtractAudio
    """
    # set path to acoustic feats
    acoustic_save_path = os.path.join(base_path, acoustic_save_dir)
    # create the save directory if it doesn't exist
    if not os.path.exists(acoustic_save_path):
        os.makedirs(acoustic_save_path)

    for audio_dir in os.listdir(base_path):
        path_to_files = os.path.join(base_path, audio_dir)
        if os.path.isdir(path_to_files):

            # extract features using opensmile
            for audio_file in os.listdir(path_to_files):
                audio_name = audio_file.split(".wav")[0]
                audio_save_name = str(audio_name) + "_" + acoustic_feature_set + ".csv"
                extractor = ExtractAudio(
                    path_to_files, audio_file, acoustic_save_path, smile_path
                )
                extractor.save_acoustic_csv(
                    feature_set=acoustic_feature_set, savename=audio_save_name
                )