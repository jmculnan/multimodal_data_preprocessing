# prepare chalearn for input into the model
import math
import os
import pickle
import json
import re
import sys
from collections import OrderedDict

import torch
from torch import nn
from torchtext.data import get_tokenizer

from utils.audio_extraction import (ExtractAudio,
                                    convert_to_wav,
                                    run_feature_extraction
                                    )
import pandas as pd

from utils.data_prep_helpers import (
    get_class_weights,
    get_gender_avgs,
    clean_up_word,
    get_max_num_acoustic_frames,
    transform_acoustic_item,
    get_acoustic_means
    )


class ChalearnPrep:
    """
    A class to prepare meld for input into a generic Dataset
    """

    def __init__(
        self,
        chalearn_path,
        acoustic_length,
        glove,
        f_end="_IS10.csv",
        use_cols=None,
        add_avging=True,
        avgd=False,
        pred_type="max_class",
        utts_file_name="gold_and_utts.tsv"
    ):
        self.path = chalearn_path
        self.train_path = chalearn_path + "/train"
        self.dev_path = chalearn_path + "/val"
        self.test_path = chalearn_path + "/test"
        self.train = f"{self.train_path}/{utts_file_name}"
        self.dev = f"{self.dev_path}/{utts_file_name}"
        self.test = f"{self.test_path}/{utts_file_name}"

        # get files containing gold labels/data
        self.train_data_file = pd.read_csv(self.train, sep="\t")
        self.dev_data_file = pd.read_csv(self.dev, sep="\t")
        self.test_data_file = pd.read_csv(self.test, sep="\t")
        # get the type of prediction
        self.pred_type = pred_type

        # get tokenizer
        self.tokenizer = get_tokenizer("basic_english")

        # get the number of acoustic features
        self.acoustic_length = acoustic_length

        # to determine whether incoming acoustic features are averaged
        self.avgd = avgd

        if f_end != "_IS10.csv":
            setname = re.search("_(.*)\.csv", f_end)
            name = setname.group(1)
            self.train_dir = name
            self.dev_dir = name
            self.test_dir = name
        else:
            self.train_dir = "IS10"
            self.dev_dir = "IS10"
            self.test_dir = "IS10"

        print("Collecting acoustic features")

        # ordered dicts of acoustic data
        self.train_dict, self.train_acoustic_lengths = make_acoustic_dict_chalearn(
            "{0}/{1}".format(self.train_path, self.train_dir),
            f_end,
            use_cols=use_cols,
            avgd=avgd,
        )
        self.train_dict = OrderedDict(self.train_dict)
        self.dev_dict, self.dev_acoustic_lengths = make_acoustic_dict_chalearn(
            "{0}/{1}".format(self.dev_path, self.dev_dir),
            f_end,
            use_cols=use_cols,
            avgd=avgd,
        )
        self.dev_dict = OrderedDict(self.dev_dict)
        self.test_dict, self.test_acoustic_lengths = make_acoustic_dict_chalearn(
            "{0}/{1}".format(self.test_path, self.test_dir),
            f_end,
            use_cols=use_cols,
            avgd=avgd,
        )
        self.test_dict = OrderedDict(self.test_dict)

        # utterance-level dict
        self.longest_utt = self.get_longest_utt_chalearn()

        # get length of longest acoustic dataframe
        self.longest_acoustic = 1500  # set to 15 seconds

        print("Finalizing acoustic organization")

        self.train_acoustic, self.train_usable_utts = make_acoustic_set_chalearn(
            self.train,
            self.train_dict,
            acoustic_length=acoustic_length,
            longest_acoustic=self.longest_acoustic,
            add_avging=add_avging,
            avgd=avgd,
        )
        del self.train_dict
        self.dev_acoustic, self.dev_usable_utts = make_acoustic_set_chalearn(
            self.dev,
            self.dev_dict,
            acoustic_length=acoustic_length,
            longest_acoustic=self.longest_acoustic,
            add_avging=add_avging,
            avgd=avgd,
        )
        del self.dev_dict
        self.test_acoustic, self.test_usable_utts = make_acoustic_set_chalearn(
            self.test,
            self.test_dict,
            acoustic_length=acoustic_length,
            longest_acoustic=self.longest_acoustic,
            add_avging=add_avging,
            avgd=avgd,
        )
        del self.test_dict

        # get utterance, speaker, y matrices for train, dev, and test sets
        (
            self.train_utts,
            self.train_genders,
            self.train_ethnicities,
            self.train_y_extr,
            self.train_y_neur,
            self.train_y_agree,
            self.train_y_openn,
            self.train_y_consc,
            self.train_y_inter,
            self.train_utt_lengths,
            self.train_audio_ids,
        ) = self.make_data_tensors(self.train_data_file, self.train_usable_utts, glove)

        (
            self.dev_utts,
            self.dev_genders,
            self.dev_ethnicities,
            self.dev_y_extr,
            self.dev_y_neur,
            self.dev_y_agree,
            self.dev_y_openn,
            self.dev_y_consc,
            self.dev_y_inter,
            self.dev_utt_lengths,
            self.dev_audio_ids,
        ) = self.make_data_tensors(self.dev_data_file, self.dev_usable_utts, glove)

        (
            self.test_utts,
            self.test_genders,
            self.test_ethnicities,
            self.test_y_extr,
            self.test_y_neur,
            self.test_y_agree,
            self.test_y_openn,
            self.test_y_consc,
            self.test_y_inter,
            self.test_utt_lengths,
            self.test_audio_ids,
        ) = self.make_data_tensors(
            self.test_data_file, self.test_usable_utts, glove
        )

        # acoustic feature normalization based on train
        print("starting acoustic means for chalearn")
        self.all_acoustic_means, self.all_acoustic_deviations = get_acoustic_means(self.train_acoustic)

        print("starting male acoustic means for chalearn")
        self.male_acoustic_means, self.male_deviations = get_gender_avgs(
            self.train_acoustic, self.train_genders, gender=1
        )

        print("starting female acoustic means for chalearn")
        self.female_acoustic_means, self.female_deviations = get_gender_avgs(
            self.train_acoustic, self.train_genders, gender=2
        )
        print("All acoustic means calculated for chalearn")

        # get the weights for chalearn personality traits
        if pred_type == "max_class":
            all_train_ys = [[self.train_y_consc[i], self.train_y_openn[i], self.train_y_agree[i],
                             self.train_y_neur[i], self.train_y_extr[i]] for i in range(len(self.train_y_consc))]
            self.train_ys = torch.tensor([item.index(max(item)) for item in all_train_ys])
            self.trait_weights = get_class_weights(self.train_ys)

        elif pred_type == "high-low" or pred_type == "binary":
            # use the mean from train partition for each
            consc_mean = torch.mean(self.train_y_consc)
            openn_mean = torch.mean(self.train_y_openn)
            agree_mean = torch.mean(self.train_y_agree)
            extr_mean = torch.mean(self.train_y_extr)
            neur_mean = torch.mean(self.train_y_neur)

            self.train_y_consc = convert_ys(self.train_y_consc, pred_type, consc_mean)
            self.train_y_openn = convert_ys(self.train_y_consc, pred_type, openn_mean)
            self.train_y_agree = convert_ys(self.train_y_agree, pred_type, agree_mean)
            self.train_y_extr = convert_ys(self.train_y_extr, pred_type, extr_mean)
            self.train_y_neur = convert_ys(self.train_y_neur, pred_type, neur_mean)

            self.dev_y_consc = convert_ys(self.dev_y_consc, pred_type, consc_mean)
            self.dev_y_openn = convert_ys(self.dev_y_openn, pred_type, openn_mean)
            self.dev_y_agree = convert_ys(self.dev_y_agree, pred_type, agree_mean)
            self.dev_y_extr = convert_ys(self.dev_y_extr, pred_type, extr_mean)
            self.dev_y_neur = convert_ys(self.dev_y_neur, pred_type, neur_mean)

            self.test_y_consc = convert_ys(self.test_y_consc, pred_type, consc_mean)
            self.test_y_openn = convert_ys(self.test_y_openn, pred_type, openn_mean)
            self.test_y_agree = convert_ys(self.test_y_agree, pred_type, agree_mean)
            self.test_y_extr = convert_ys(self.test_y_extr, pred_type, extr_mean)
            self.test_y_neur = convert_ys(self.test_y_neur, pred_type, neur_mean)

            # get weights for each trait
            self.consc_weights = get_class_weights(self.train_y_consc)
            self.openn_weights = get_class_weights(self.train_y_openn)
            self.agree_weights = get_class_weights(self.train_y_agree)
            self.extr_weights = get_class_weights(self.train_y_extr)
            self.neur_weights = get_class_weights(self.train_y_neur)

        elif pred_type == "high-med-low" or pred_type == "ternary":
            # get the locations you want
            # here we want 1/3 of the data and 2/3 of the data
            q_measure = torch.tensor([.33, .67])

            # get the 1/3 and 2/3 quantiles
            consc_onethird, consc_twothird = torch.quantile(self.train_y_consc, q_measure)
            openn_onethird, openn_twothird = torch.quantile(self.train_y_openn, q_measure)
            agree_onethird, agree_twothird = torch.quantile(self.train_y_agree, q_measure)
            extr_onethird, extr_twothird = torch.quantile(self.train_y_extr, q_measure)
            neur_onethird, neur_twothird = torch.quantile(self.train_y_neur, q_measure)

            self.train_y_consc = convert_ys(self.train_y_consc, pred_type, one_third=consc_onethird,
                                            two_thirds=consc_twothird)
            self.train_y_openn = convert_ys(self.train_y_consc, pred_type, one_third=openn_onethird,
                                            two_thirds=openn_twothird)
            self.train_y_agree = convert_ys(self.train_y_agree, pred_type, one_third=agree_onethird,
                                            two_thirds=agree_twothird)
            self.train_y_extr = convert_ys(self.train_y_extr, pred_type, one_third=extr_onethird,
                                           two_thirds=extr_twothird)
            self.train_y_neur = convert_ys(self.train_y_neur, pred_type, one_third=neur_onethird,
                                           two_thirds=neur_twothird)

            self.dev_y_consc = convert_ys(self.dev_y_consc, pred_type, one_third=consc_onethird,
                                            two_thirds=consc_twothird)
            self.dev_y_openn = convert_ys(self.dev_y_openn, pred_type, one_third=openn_onethird,
                                            two_thirds=openn_twothird)
            self.dev_y_agree = convert_ys(self.dev_y_agree, pred_type, one_third=agree_onethird,
                                            two_thirds=agree_twothird)
            self.dev_y_extr = convert_ys(self.dev_y_extr, pred_type, one_third=extr_onethird,
                                           two_thirds=extr_twothird)
            self.dev_y_neur = convert_ys(self.dev_y_neur, pred_type, one_third=neur_onethird,
                                           two_thirds=neur_twothird)

            self.test_y_consc = convert_ys(self.test_y_consc, pred_type, one_third=consc_onethird,
                                            two_thirds=consc_twothird)
            self.test_y_openn = convert_ys(self.test_y_openn, pred_type, one_third=openn_onethird,
                                            two_thirds=openn_twothird)
            self.test_y_agree = convert_ys(self.test_y_agree, pred_type, one_third=agree_onethird,
                                            two_thirds=agree_twothird)
            self.test_y_extr = convert_ys(self.test_y_extr, pred_type, one_third=extr_onethird,
                                           two_thirds=extr_twothird)
            self.test_y_neur = convert_ys(self.test_y_neur, pred_type, one_third=neur_onethird,
                                           two_thirds=neur_twothird)

            # get weights for each trait
            self.consc_weights = get_class_weights(self.train_y_consc)
            self.openn_weights = get_class_weights(self.train_y_openn)
            self.agree_weights = get_class_weights(self.train_y_agree)
            self.extr_weights = get_class_weights(self.train_y_extr)
            self.neur_weights = get_class_weights(self.train_y_neur)

        # get the data organized for input into the NNs
        # self.train_data, self.dev_data, self.test_data = self.combine_xs_and_ys()
        self.train_data, self.dev_data, self.test_data = self.combine_xs_and_ys()

    def combine_xs_and_ys(self):
        """
        Combine all x and y data into list of tuples for easier access with DataLoader
        """
        train_data = []
        dev_data = []
        test_data = []

        for i, item in enumerate(self.train_acoustic):
            # normalize
            # if self.train_genders[i] == 2:
            #     item_transformed = transform_acoustic_item(
            #         item, self.female_acoustic_means, self.female_deviations
            #     )
            # else:
            #     item_transformed = transform_acoustic_item(
            #         item, self.male_acoustic_means, self.male_deviations
            #     )
            item_transformed = transform_acoustic_item(
                item, self.all_acoustic_means, self.all_acoustic_deviations
            )
            if self.pred_type is not "max_class":
                train_data.append(
                    (
                        item_transformed,
                        self.train_utts[i],
                        0,  # todo: eventually add speaker ?
                        self.train_genders[i],
                        self.train_ethnicities[i],
                        self.train_y_extr[i],
                        self.train_y_neur[i],
                        self.train_y_agree[i],
                        self.train_y_openn[i],
                        self.train_y_consc[i],
                        self.train_y_inter[i],
                        self.train_audio_ids[i],
                        self.train_utt_lengths[i],
                        self.train_acoustic_lengths[i],
                    )
                )
            else:
                ys = [self.train_y_extr[i], self.train_y_neur[i],
                      self.train_y_agree[i], self.train_y_openn[i],
                      self.train_y_consc[i]]
                item_y = ys.index(max(ys))
                train_data.append((
                    item_transformed,
                    self.train_utts[i],
                    0,
                    self.train_genders[i],
                    torch.tensor(item_y),
                    self.train_ethnicities[i],
                    self.train_audio_ids[i],
                    self.train_utt_lengths[i],
                    self.train_acoustic_lengths[i]
                    )
                )

        for i, item in enumerate(self.dev_acoustic):
            # if self.dev_genders[i] == 2:
            #     item_transformed = transform_acoustic_item(
            #         item, self.female_acoustic_means, self.female_deviations
            #     )
            # else:
            #     item_transformed = transform_acoustic_item(
            #         item, self.male_acoustic_means, self.male_deviations
            #     )
            item_transformed = transform_acoustic_item(
                item, self.all_acoustic_means, self.all_acoustic_deviations
            )
            if self.pred_type is not "max_class":
                dev_data.append(
                    (
                        item_transformed,
                        self.dev_utts[i],
                        0,  # todo: eventually add speaker ?
                        self.dev_genders[i],
                        self.dev_ethnicities[i],
                        self.dev_y_extr[i],
                        self.dev_y_neur[i],
                        self.dev_y_agree[i],
                        self.dev_y_openn[i],
                        self.dev_y_consc[i],
                        self.dev_y_inter[i],
                        self.dev_audio_ids[i],
                        self.dev_utt_lengths[i],
                        self.dev_acoustic_lengths[i],
                    )
                )
            else:
                ys = [self.dev_y_extr[i], self.dev_y_neur[i],
                      self.dev_y_agree[i], self.dev_y_openn[i],
                      self.dev_y_consc[i]]
                item_y = ys.index(max(ys))
                dev_data.append(
                    (
                        item_transformed,
                        self.dev_utts[i],
                        0,  # todo: eventually add speaker ?
                        self.dev_genders[i],
                        torch.tensor(item_y),
                        self.dev_ethnicities[i],
                        self.dev_audio_ids[i],
                        self.dev_utt_lengths[i],
                        self.dev_acoustic_lengths[i],
                    )
                )

        for i, item in enumerate(self.test_acoustic):
            # if self.test_genders[i] == 2:
            #     item_transformed = transform_acoustic_item(
            #         item, self.female_acoustic_means, self.female_deviations
            #     )
            # else:
            #     item_transformed = transform_acoustic_item(
            #         item, self.male_acoustic_means, self.male_deviations
            #     )
            item_transformed = transform_acoustic_item(
                item, self.all_acoustic_means, self.all_acoustic_deviations
            )
            if self.pred_type is not "max_class":
                test_data.append(
                    (
                        item_transformed,
                        self.test_utts[i],
                        0,  # todo: eventually add speaker ?
                        self.test_genders[i],
                        self.test_ethnicities[i],
                        self.test_y_extr[i],
                        self.test_y_neur[i],
                        self.test_y_agree[i],
                        self.test_y_openn[i],
                        self.test_y_consc[i],
                        self.test_y_inter[i],
                        self.test_audio_ids[i],
                        self.test_utt_lengths[i],
                        self.test_acoustic_lengths[i],
                    )
                )
            else:
                ys = [self.test_y_extr[i], self.test_y_neur[i],
                      self.test_y_agree[i], self.test_y_openn[i],
                      self.test_y_consc[i]]
                item_y = ys.index(max(ys))
                test_data.append(
                    (
                        item_transformed,
                        self.test_utts[i],
                        0,  # todo: eventually add speaker ?
                        self.test_genders[i],
                        torch.tensor(item_y),
                        self.test_ethnicities[i],
                        self.test_audio_ids[i],
                        self.test_utt_lengths[i],
                        self.test_acoustic_lengths[i],
                    )
                )

        return train_data, dev_data, test_data

    def get_longest_utt_chalearn(self):
        """
        Get the length of the longest utterance and dialogue in the meld
        :return: length of longest utt, length of longest dialogue
        """
        longest = 0

        # get all data splits
        train_utts_df = self.train_data_file
        dev_utts_df = self.dev_data_file
        # test_utts_df = self.test_data_file

        # concatenate them and put utterances in array
        # all_utts_df = pd.concat([train_utts_df, dev_utts_df, test_utts_df], axis=0)
        all_utts_df = pd.concat([train_utts_df, dev_utts_df], axis=0)

        all_utts = all_utts_df["utterance"].tolist()

        for i, item in enumerate(all_utts):
            try:
                item = clean_up_word(item)
            except AttributeError:  # at least one item is blank and reads in as a float
                item = "<UNK>"
            item = self.tokenizer(item)
            if len(item) > longest:
                longest = len(item)

        return longest

    def make_data_tensors(self, all_utts_df, all_utts_list, glove):
        """
        Prepare the tensors of utterances + genders, gold labels
        :param all_utts_df: the df containing the text (in column 0)
        :param all_utts_list: a list of all usable utterances
        :param glove: an instance of class Glove
        :return:
        """
        # create holders for the data
        all_utts = []
        all_genders = []
        all_ethnicities = []
        all_extraversion = []
        all_neuroticism = []
        all_agreeableness = []
        all_openness = []
        all_conscientiousness = []
        all_interview = []
        all_audio_ids = []

        # create holder for sequence lengths information
        utt_lengths = []

        for idx, row in all_utts_df.iterrows():

            # check to make sure this utterance is used
            audio_name = row["file"]
            audio_id = audio_name.split(".mp4")[0]
            if audio_id in all_utts_list:
                # add audio id to list
                all_audio_ids.append(audio_id)

                # create utterance-level holders
                utts = [0] * self.longest_utt

                # get values from row
                try:
                    utt = clean_up_word(row["utterance"])
                except AttributeError:  # at least one item is blank and reads in as a float
                    utt = "<UNK>"
                utt = self.tokenizer(utt)
                utt_lengths.append(len(utt))

                gen = row["gender"]
                eth = row["ethnicity"]
                extra = row["extraversion"]
                neur = row["neuroticism"]
                agree = row["agreeableness"]
                openn = row["openness"]
                consc = row["conscientiousness"]
                inter = row["invite_to_interview"]

                # convert words to indices for glove
                utt_indexed = glove.index(utt)
                for i, item in enumerate(utt_indexed):
                    utts[i] = item

                all_utts.append(torch.tensor(utts))
                all_genders.append(gen)
                all_ethnicities.append(eth)
                all_extraversion.append(extra)
                all_neuroticism.append(neur)
                all_agreeableness.append(agree)
                all_openness.append(openn)
                all_conscientiousness.append(consc)
                all_interview.append(inter)

        # create pytorch tensors for each
        all_genders = torch.tensor(all_genders)
        all_ethnicities = torch.tensor(all_ethnicities)
        all_extraversion = torch.tensor(all_extraversion)
        all_neuroticism = torch.tensor(all_neuroticism)
        all_agreeableness = torch.tensor(all_agreeableness)
        all_openness = torch.tensor(all_openness)
        all_conscientiousness = torch.tensor(all_conscientiousness)
        all_interview = torch.tensor(all_interview)

        # return data
        return (
            all_utts,
            all_genders,
            all_ethnicities,
            all_extraversion,
            all_neuroticism,
            all_agreeableness,
            all_openness,
            all_conscientiousness,
            all_interview,
            utt_lengths,
            all_audio_ids
        )


def convert_chalearn_pickle_to_json(path, file):
    """
    Convert the pickled data files for chalearn into json files
    """
    fname = file.split(".pkl")[0]
    pickle_file = os.path.join(path, file)
    with open(pickle_file, "rb") as pfile:
        # use latin-1 enecoding to avoid readability issues
        data = pickle.load(pfile, encoding="latin1")

    json_file = os.path.join(path, fname + ".json")
    with open(json_file, "w") as jfile:
        json.dump(data, jfile)


def convert_ys(ys, conversion="high-low", mean_y=None,
               one_third=None, two_thirds=None):
    """
    Convert a set of ys into binary high-low labels
    or ternary high-medium-low labels
    Uses the mean for binary and one-third and two-third
        markers for ternary labels
    """
    new_ys = []
    if conversion == "high-low" or conversion == "binary":
        for item in ys:
            if mean_y:
                if item >= mean_y:
                    new_ys.append(1)
                else:
                    new_ys.append(0)
            else:
                if item >= 0.5:
                    new_ys.append(1)
                else:
                    new_ys.append(0)
    elif conversion == "high-med-low" or conversion == "ternary":
        if one_third and two_thirds:
            for item in ys:
                if item >= two_thirds:
                    new_ys.append(2)
                elif one_third <= item < two_thirds:
                    new_ys.append(1)
                else:
                    new_ys.append(0)
        else:
            for item in ys:
                if item >= 0.67:
                    new_ys.append(2)
                elif 0.34 <= item < 0.67:
                    new_ys.append(1)
                else:
                    new_ys.append(0)
    return new_ys


def preprocess_chalearn_data(
    base_path, acoustic_save_dir, smile_path, acoustic_feature_set="IS10"
):
    """
    Preprocess the ravdess data by extracting acoustic features from wav files
    base_path : the path to the base RAVDESS directory
    paths : list of paths where audio is located
    acoustic_save_dir : the directory in which to save acoustic feature files
    smile_path : the path to OpenSMILE
    acoustic_feature_set : the feature set to use with ExtractAudio
    """
    path_to_train = os.path.join(base_path, "train")
    path_to_dev = os.path.join(base_path, "val")
    path_to_test = os.path.join(base_path, "test")
    # paths = [path_to_train, path_to_dev, path_to_test]
    paths = [path_to_test]

    for p in paths:
        # set path to audio files
        path_to_files = os.path.join(p, "mp4")
        # set path to acoustic feats
        acoustic_save_path = os.path.join(p, acoustic_save_dir)
        # create the save directory if it doesn't exist
        if not os.path.exists(acoustic_save_path):
            os.makedirs(acoustic_save_path)

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


def create_gold_tsv_chalearn(gold_file, utts_file, gender_file, save_name):
    """
    Create the gold tsv file from a json file
    gold_file : path to JSON file containing y values
    utts_file : path to JSON transcription file
    gender_file : the path to a CSV file with gender info
    """
    with open(gold_file, "r") as jfile:
        data = json.load(jfile)

    with open(utts_file, "r") as j2file:
        utts_dict = json.load(j2file)

    genders = pd.read_csv(gender_file, sep=";")
    genders_dict = dict(zip(genders["VideoName"], genders["Gender"]))
    ethnicity_dict = dict(zip(genders["VideoName"], genders["Ethnicity"]))

    all_files_utts = sorted(utts_dict.keys())
    all_files = sorted(data["extraversion"].keys())

    if all_files != all_files_utts:
        print("utts dict and gold labels don't contain the same set of files")
    if all_files != sorted(genders_dict.keys()):
        print("gold labels and gender labels don't contain the same set of files")

    with open(save_name, "w") as tsvfile:
        tsvfile.write(
            "file\tgender\tethnicity\textraversion\tneuroticism\t"
            "agreeableness\topenness\tconscientiousness\tinvite_to_interview\tutterance\n"
        )
        for item in all_files:
            gender = genders_dict[item]
            ethnicity = ethnicity_dict[item]
            extraversion = data["extraversion"][item]
            neuroticism = data["neuroticism"][item]
            agreeableness = data["agreeableness"][item]
            openness = data["openness"][item]
            conscientiousness = data["conscientiousness"][item]
            interview = data["interview"][item]
            try:
                utterance = utts_dict[item]
            except KeyError:
                utterance = ""
            tsvfile.write(
                f"{item}\t{gender}\t{ethnicity}\t{extraversion}\t{neuroticism}\t{agreeableness}\t"
                f"{openness}\t{conscientiousness}\t{interview}\t{utterance}\n"
            )


def make_acoustic_dict_chalearn(
    acoustic_path, f_end="_IS10.csv", use_cols=None, avgd=True
):
    """
    makes a dict of clip_id: data for use in MELD objects
    f_end: end of acoustic file names
    use_cols: if set, should be a list [] of column names to include
    n_to_skip : the number of columns at the start to ignore (e.g. name, time)
    """
    acoustic_dict = {}
    # acoustic_lengths = []
    acoustic_lengths = {}
    # find acoustic features files
    for f in os.listdir(acoustic_path):
        if f.endswith(f_end):
            # set the separator--non-averaged files are ;SV
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

            # get the dialogue and utterance IDs
            id = f.split(f_end)[0]

            # save the dataframe to a dict with (dialogue, utt) as key
            if feats.shape[0] > 0:
                acoustic_dict[id] = feats.values.tolist()
                acoustic_lengths[id] = feats.shape[0]

            # delete the features df bc it takes up masses of space
            del feats

    # sort acoustic lengths so they are in the same order as other data
    acoustic_lengths = [value for key, value in sorted(acoustic_lengths.items())]

    return acoustic_dict, acoustic_lengths


def make_acoustic_set_chalearn(
    text_path,
    acoustic_dict,
    acoustic_length,
    longest_acoustic,
    add_avging=True,
    avgd=False,
):
    """
    Prep the acoustic data using the acoustic dict
    :param text_path: FULL path to file containing utterances + labels
    :param acoustic_dict:
    :param add_avging: whether to average the feature sets
    :return:
    """
    # read in the acoustic csv
    if type(text_path) == str:
        all_utts_df = pd.read_csv(text_path, sep="\t")
    elif type(text_path) == pd.core.frame.DataFrame:
        all_utts_df = text_path
    else:
        sys.exit("text_path is of unaccepted type.")

    # get lists of valid dialogues and utterances
    valid_utts = all_utts_df["file"].tolist()

    # set holders for acoustic data
    all_acoustic = []
    usable_utts = []

    # for all items with audio + gold label
    for idx, item in enumerate(valid_utts):
        item_id = item.split(".mp4")[0]
        # if that dialogue and utterance appears has an acoustic feats file
        if item_id in acoustic_dict.keys():

            # pull out the acoustic feats dataframe
            acoustic_data = acoustic_dict[item_id]

            # add this dialogue + utt combo to the list of possible ones
            usable_utts.append(item_id)

            if not avgd and not add_avging:
                # set intermediate acoustic holder
                acoustic_holder = torch.zeros((longest_acoustic, acoustic_length))

                # add the acoustic features to the holder of features
                for i, feats in enumerate(acoustic_data):

                    # for now, using longest acoustic file in TRAIN only
                    if i >= longest_acoustic:
                        break
                    # needed because some files allegedly had length 0
                    for j, feat in enumerate(feats):
                        acoustic_holder[i][j] = feat
            else:
                if avgd:
                    acoustic_holder = torch.tensor(acoustic_data)
                elif add_avging:
                    data_len = len(acoustic_data)
                    # skip first and last 25%
                    acoustic_holder = torch.mean(torch.tensor(acoustic_data)[math.floor(data_len * 0.25):math.ceil(data_len * 0.75)], dim=0)

            # add features as tensor to acoustic data
            all_acoustic.append(acoustic_holder)

    # pad the sequence and reshape it to proper format
    # this is here to keep the formatting for acoustic RNN
    all_acoustic = nn.utils.rnn.pad_sequence(all_acoustic)
    all_acoustic = all_acoustic.transpose(0, 1)

    return all_acoustic, usable_utts


def reorganize_gender_annotations_chalearn(path, genderfile, transcriptfile):
    """
    Use the file containing transcriptions of utterances to compare files
    and separate gender annotation val file into 2 csv files
    path : the path to the dataset
    genderfile : the path to the file containing the val gender key
    transcriptfile : the path to the file containing transcriptions for val set
    """
    # get gender dataframe
    genf = pd.read_csv(genderfile, sep=";")

    # get transcription json
    with open(transcriptfile, "r") as tfile:
        dev_data = json.load(tfile)

    # get list of files in dev set
    dev_files = dev_data.keys()

    print(len(dev_files))

    # get gender dataframes for dev and train
    genf_dev = genf[genf["VideoName"].isin(dev_files)]
    genf_train = genf[~genf["VideoName"].isin(dev_files)]

    print(genf.shape)
    print(genf_train.shape)
    print(genf_dev.shape)

    val_saver = os.path.join(path, "val/gender_annotations_val.csv")
    train_saver = os.path.join(path, "train/gender_annotations_train.csv")

    genf_dev.to_csv(val_saver, index=False)
    genf_train.to_csv(train_saver, index=False)


if __name__ == "__main__":
    # path to data
    path = "../../datasets/multimodal_datasets/Chalearn"
    # train_path = os.path.join(path, "train/mp4")
    # val_path = os.path.join(path, "val/mp4")
    test_path = os.path.join(path, "test/mp4")
    # pathd = [train_path, val_path]
    paths = [test_path]
    # path to opensmile
    smile_path = "../../opensmile-2.3.0"
    # acoustic set to extract
    acoustic_set = "IS10"

    # #### WHEN RUNNING FOR THE FIRST TIME
    # # 1. convert pickle to json
    # file_1 = "test/annotation_test.pkl"
    # file_2 = "test/transcription_test.pkl"
    #
    # convert_chalearn_pickle_to_json(path, file_1)
    # convert_chalearn_pickle_to_json(path, file_2)
    #
    # # 2. convert mp4 to wav
    # # for p in paths:
    # #     print(p)
    # #     print("====================================")
    # #     for f in os.listdir(p):
    # #         if f.endswith(".mp4"):
    # #             print(f)
    # #             convert_to_wav(os.path.join(p, f))
    #
    # # 3. preprocess files
    # # preprocess_chalearn_data(path, "IS10", smile_path, acoustic_set)
    #
    # # 3.1 reorganize gender annotations file
    # gender_file = os.path.join(path, "test/gender_anntoations_test.csv")
    # anno_file = os.path.join(path, "test/transcription_test.json")
    # reorganize_gender_annotations_chalearn(path, gender_file, anno_file)
    #
    # # 4. create a gold CSV to examine data more closely by hand
    # test_gold_json_path = os.path.join(path, "test/annotation_test.json")
    # test_utts_json_path = os.path.join(path, "test/transcription_test.json")
    # test_gender_file = os.path.join(path, "test/gender_anntoations_test.csv")
    # test_save_name = os.path.join(path, "test/gold_and_utts.tsv")
    # create_gold_tsv_chalearn(
    #     test_gold_json_path, test_utts_json_path, test_gender_file, test_save_name
    # )
    # #
    # # dev_gold_json_path = os.path.join(path, "val/annotation_validation.json")
    # # dev_utts_json_path = os.path.join(path, "val/transcription_validation.json")
    # # dev_gender_file = os.path.join(path, "val/gender_annotations_val.csv")
    # # dev_save_name = os.path.join(path, "val/gold_and_utts.tsv")
    # # create_gold_tsv_chalearn(
    # #     dev_gold_json_path, dev_utts_json_path, dev_gender_file, dev_save_name
    # # )
    # #
    # # train_gold_json_path = os.path.join(path, "train/annotation_training.json")
    # # train_utts_json_path = os.path.join(path, "train/transcription_training.json")
    # # train_gender_file = os.path.join(path, "train/gender_annotations_train.csv")
    # # train_save_name = os.path.join(path, "train/gold_and_utts.tsv")
    # # create_gold_tsv_chalearn(
    # #     train_gold_json_path, train_utts_json_path, train_gender_file, train_save_name
    # # )

    #### TO CREATE ADDITIONAL ACOUSTIC FEATURE SETS
    train_audio_path = os.path.join(path, "train/wav")
    dev_audio_path = os.path.join(path, "val/wav")
    test_audio_path = os.path.join(path, "test/wav")

    train_save_dir = os.path.join(path, "train/IS11")
    dev_save_dir = os.path.join(path, "val/IS11")
    test_save_dir = os.path.join(path, "test/IS11")

    run_feature_extraction(train_audio_path, "IS11", train_save_dir)
    run_feature_extraction(dev_audio_path, "IS11", dev_save_dir)
    run_feature_extraction(test_audio_path, "IS11", test_save_dir)
