
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.utils import compute_class_weight

from collections import OrderedDict
import pandas as pd
import numpy as np
import functools
import operator
import random

from utils.data_prep_helpers import (
    MinMaxScaleRange,
    clean_up_word,
    get_avg_vec,
    scale_feature,
    get_data_samples,
    transform_acoustic_item, make_glove_dict, Glove,
)
from prep_data import *

def prep_asist_data(
    data_path="../../asist_data2/overall_sent-emo.csv",
    feature_set="IS13",
    transcription_type="gold",
    embedding_type="distilbert",
    glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
    features_to_use=None,
    as_dict=False,
    avg_acoustic_data=False,
    custom_feats_file=None,
    num_train_ex=None,
    include_spectrograms=False,
):
    # load glove
    if embedding_type.lower() == "glove":
        glove_dict = make_glove_dict(glove_filepath)
        glove = Glove(glove_dict)
    else:
        glove = None

    # holder for name of file containing utterance info
    utts_name = "overall_sent-emo.csv"

    # create instance of StandardPrep class
    asist_prep = SelfSplitPrep(
        data_type="asist",
        data_path=data_path,
        feature_set=feature_set,
        utterance_fname=utts_name,
        glove=glove,
        use_cols=features_to_use,
        as_dict=as_dict,
        avg_acoustic_data=avg_acoustic_data,
        custom_feats_file=custom_feats_file,
        bert_type=embedding_type,
        include_spectrograms=include_spectrograms
    )

    # get train, dev, test data
    train_data, dev_data, test_data = asist_prep.get_data_folds()

    # get train ys
    train_ys = [item['ys'] for item in train_data]

    # get updated class weights using train ys
    class_weights = get_updated_class_weights_multicat(train_ys)

    return train_data, dev_data, test_data, class_weights


def get_updated_class_weights_multicat(train_ys):
    """
    Get updated class weights
    Because DataPrep assumes you only enter train set
    :return:
    """
    all_task_weights = []
    # get class weights for each task
    for i in range(len(train_ys[0])):
        labels = [int(y[i]) for y in train_ys]
        classes = sorted(list(set(labels)))
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        weights = torch.tensor(weights, dtype=torch.float)
        all_task_weights.append(weights)

    return all_task_weights

class AsistDataset(Dataset):
    def __init__(
        self,
        acoustic_dict,
        glove,
        ys_path=None,
        splits=3,
        norm="minmax",
        sequence_prep=None,
        truncate_from="start",
        add_avging=False,
        transcript_type="zoom",
    ):
        """
        :param acoustic_dict: dict of {utt_id : data}
        :param glove: an instance of class Glove
        :param ys_path: path to dataframe of sid + ys
        :param splits: number of splits for CV
        :param norm: the type of data normalization
        :param sequence_prep: the way sequences are handled, options: truncate, pad, None
        :param truncate_from: whether to truncate from start or end of file
        """
        self.cols_to_skip = 9 if transcript_type.lower() == "zoom" else 10
        self.acoustic_dict = OrderedDict(
            {
                key: df
                for key, df in acoustic_dict.items()
            }
        )
        self.glove = glove
        if ys_path is not None:
            self.ys_df = pd.read_csv(ys_path)
            self.valid_files = self.ys_df["sid"].tolist()
        else:
            self.ys_df = None
            self.valid_files = [key for key in self.acoustic_dict.keys()]
        self.norm = norm
        self.sequence_prep = sequence_prep
        self.truncate_from = truncate_from

        self.add_avging = add_avging

        if norm == "minmax":
            # currently uses index-keyed min-max for features
            # should we change it later?
            self.min_max_scaler = MinMaxScaleRange()
            self.get_min_max_scales()

        self.skipped_files = []
        # Tests gender classifier:print('start')

        # self.x_acoustic, self.x_glove, self.x_speaker, self.x_utt_lengths = self.combine_acoustic_and_glove()
        # Tests gender classifierprint('end')
        if transcript_type.lower() == "zoom":
            (
                self.x_acoustic,
                self.x_glove,
                self.x_speaker,
                self.x_utt_lengths,
            ) = self.combine_acoustic_and_glove_utt_level()
        else:
            (
                self.x_acoustic,
                self.x_glove,
                self.x_speaker,
                self.x_utt_lengths,
                # self.x_speaker_gender, #sa
            ) = self.combine_acoustic_and_glove_wd_level()

        # todo: we should get gender info on participants OR predict it
        # add call to wrapper function that calls the gender classifier
        self.speaker_gender_data = 0
        if self.ys_df is not None:
            self.y_data = self.create_ordered_ys()
            self.y_data = self.create_ordered_ys_utt_level(
                num_utts=len(self.x_utt_lengths)
            )
        self.data = self.combine_data()
        self.splits = splits
        self.data_for_model_input = self.get_data_splits()

        # for working with an individual split
        self.current_split = []
        self.current_split_num = 0
        self.val_split = []
        self.remaining_splits = []

        self.set_split(0)

    def get_min_max_scales(self):
        for df in self.acoustic_dict.values():

            for row in df_subset.itertuples():
                for i, wd in enumerate(row):
                    if i >= self.cols_to_skip + 1:
                        self.min_max_scaler.update((i - (self.cols_to_skip + 1)), wd)

    def __len__(self):
        return len(self.current_split)

    def __getitem__(self, index):
        return self.current_split[index]

    def set_split(self, n):
        # set the split; n is the TEST split, n-1 is DEV split
        self.current_split = self.data_for_model_input[n]
        self.current_split_num = n
        if n - 1 >= 0:
            prev = n - 1
        else:
            prev = max(self.data_for_model_input.keys())
        self.val_split = self.data_for_model_input[prev]
        remaining_splits = [
            val
            for key, val in self.data_for_model_input.items()
            if (key != n and key != prev)
        ]
        self.remaining_splits = functools.reduce(operator.iconcat, remaining_splits, [])

    def get_data_splits(self):
        data_dict = {}
        # calculate length of data
        total_len = len(self.data)
        # calculate length of each split based on number
        split_len = int(total_len / self.splits)
        # random permutation of indices
        indices = np.random.permutation(total_len)
        # get split indices
        c = 0
        # for all but final split
        for split in range(self.splits - 1):
            split_data = []
            split_idx = indices[split_len * c : split_len * (c + 1)]
            # get data for each of the indices
            for i in split_idx:
                split_data.append(self.data[i])
            # add to dict of splits
            data_dict[c] = split_data
            c += 1
        # for final split do the same
        split_idx = indices[split_len * c :]
        split_data = []
        for i in split_idx:
            split_data.append(self.data[i])
        data_dict[c] = split_data
        return data_dict

    def combine_acoustic_and_glove_utt_level(self):
        """
        Combine acoustic feats + glove indices (speaker info, too)
        :return: list of double of ([acoustic], [glove_index], [speaker])
        This makes each utterance its own data point; does NOT use conversational structure
        Doing this so that the data may be put into existing networks 8.27
        """
        print("Data prep and normalization starting")

        # set holders for acoustic, words, and speakers
        acoustic_data = []
        ordered_words = []
        ordered_speakers = []

        utt_lengths = []

        # skip the the first cols_to_skip columns
        start_idx = self.cols_to_skip

        # counter for smallest dataframe for truncation
        # get skipped files based on number of items
        #   e.g. too few utts
        if self.sequence_prep == "truncate":
            smallest = self.truncate_seq()

        # get the longest utterance

        longest_utt = get_longest_utterance_asist(
            [
                item
                for key, item in self.acoustic_dict.items()
                if key in self.valid_files
            ]
        )

        speaker_list = []
        for key, item in self.acoustic_dict.items():
            if key in self.valid_files:
                speakers = set(item["speaker"])
                speaker_list.extend([str(item) for item in speakers])

        all_speakers = sorted(list(set(speaker_list)))

        # iterate through items in the acoustic dict
        for key, item in self.acoustic_dict.items():
            # if the item has gold data
            if key in self.valid_files:

                # for each row in that item's dataframe
                for idx, row in item.iterrows():
                    # get the speaker
                    spkr = str(row["speaker"])

                    ordered_speakers.append(all_speakers.index(spkr))

                    # get the word
                    utt = clean_up_word(row["utt"]).lower()
                    utt_wds = [0] * longest_utt
                    wds = [wd for wd in utt.strip().split(" ")]
                    utt_lengths.append(len(wds))
                    for i, wd in enumerate(wds):
                        # save that word's index
                        if wd in self.glove.wd2idx.keys():
                            utt_wds[i] = self.glove.wd2idx[wd]
                        else:
                            utt_wds[i] = self.glove.wd2idx["<UNK>"]

                    # save the acoustic information in remaining columns
                    # todo: verify this works
                    row_vals = row.values[:-start_idx].tolist()

                    if self.sequence_prep == "truncate":
                        ordered_words.append(utt_wds)
                        acoustic_data.append(row_vals)
                    else:
                        ordered_words.append(torch.tensor(utt_wds))
                        acoustic_data.append(torch.tensor(row_vals))

                    # if using min-max scaling, scale the data
                    if self.norm == "minmax":
                        self.minmax_scale(row_vals, lower=0, upper=1)

        # use zero-padding to make all sequences the same length
        # if we need to pad, we MUST pack
        if self.sequence_prep == "pad":
            acoustic_data = nn.utils.rnn.pad_sequence(acoustic_data)
            ordered_words = nn.utils.rnn.pad_sequence(ordered_words)

            # swap axes to get (total_inputs, length_of_sequence, length_of_vector)
            acoustic_data = acoustic_data.transpose(0, 1)
            ordered_words = ordered_words.transpose(0, 1)

        elif self.sequence_prep == "truncate":
            if self.truncate_from == "start":
                acoustic_data = [item[-smallest:] for item in acoustic_data]
                ordered_words = [item[-smallest:] for item in ordered_words]
            else:
                acoustic_data = [item[:smallest] for item in acoustic_data]
                ordered_words = [item[:smallest] for item in ordered_words]

            acoustic_data = torch.tensor(acoustic_data)
            ordered_words = torch.tensor(ordered_words)

        print("Acoustic data size is: " + str(acoustic_data.shape))
        print("Ordered words is: " + str(ordered_words.shape))
        print("Ordered speakers size is: " + str(len(ordered_speakers)))
        print("Utterance lengths size is: " + str(len(utt_lengths)))

        print("Data prep and normalization complete")

        # return acoustic info, words indices, speaker
        return acoustic_data, ordered_words, ordered_speakers, utt_lengths

    def combine_acoustic_and_glove_wd_level(self):
        """
        Prepare the data when it is word-level aligned
        """
        print("Data prep and normalization starting")

        # set holders for acoustic, words, and speakers
        acoustic_data = []
        ordered_words = []
        ordered_speakers = []

        utt_lengths = []

        # skip the the first cols_to_skip columns
        start_idx = self.cols_to_skip

        # counter for smallest dataframe for truncation
        # get skipped files based on number of items
        #   e.g. too few utts
        if self.sequence_prep == "truncate":
            smallest = self.truncate_seq()

        # get the longest utterance
        longest_utt = get_longest_aws_utterance_asist(
            [
                item
                for key, item in self.acoustic_dict.items()
                if key in self.valid_files
            ]
        )

        speaker_list = []
        for key, item in self.acoustic_dict.items():
            if key in self.valid_files:
                speakers = set(item["speaker"])
                speaker_list.extend([str(item) for item in speakers])

        all_speakers = sorted(list(set(speaker_list)))

        print(self.valid_files)
        # iterate through items in the acoustic dict
        for key, item in self.acoustic_dict.items():
            print(f"key is: {key}")
            # if the item has gold data
            if key in self.valid_files:
                # set holders
                utt_wds = [0] * longest_utt
                utt_acoustic = []

                utt_num = 0
                wd_in_utt = 0
                spkr = None

                # for each row in that item's dataframe
                for idx, row in item.iterrows():
                    # get acoustic data
                    row_vals = row.values[start_idx:].tolist()

                    # get the word
                    wd = clean_up_word(row["word"]).lower()
                    if wd in self.glove.wd2idx.keys():
                        wd_idx = self.glove.wd2idx[wd]
                    else:
                        wd_idx = self.glove.wd2idx["<UNK>"]
                    # if we are still in the same utterance
                    if utt_num == row["utt_num"]:
                        # add the word
                        utt_wds[wd_in_utt] = wd_idx
                        wd_in_utt += 1
                        # get the speaker
                        spkr = str(row["speaker"])
                        # add acoustic features to holder
                        utt_acoustic.append(row_vals)
                    else:
                        # reset the holder
                        if self.sequence_prep == "truncate":
                            ordered_words.append(utt_wds)
                        else:
                            ordered_words.append(torch.tensor(utt_wds))
                        # average acoustic data, if needed
                        if self.add_avging:
                            try:
                                utt_acoustic = get_avg_vec(utt_acoustic)
                            except TypeError:
                                # just keep it the same if it's a flat list
                                utt_acoustic = utt_acoustic
                        # add aoustic data to list
                        if self.sequence_prep == "truncate":
                            acoustic_data.append(utt_acoustic)
                        else:
                            acoustic_data.append(torch.tensor(utt_acoustic))
                        utt_wds = [0] * longest_utt
                        # get the length of the utterance
                        utt_lengths.append(wd_in_utt + 1)
                        wd_in_utt = 0
                        # add the new word
                        utt_wds[0] = wd_idx
                        # append the speaker
                        ordered_speakers.append(all_speakers.index(spkr))
                        # update speaker
                        spkr = str(row["speaker"])

                    # # if using min-max scaling, scale the data
                    # todo: reimplement me
                    # if self.norm == "minmax":
                    #     self.minmax_scale(row_vals, lower=0, upper=1)

        # use zero-padding to make all sequences the same length
        # if we need to pad, we MUST pack
        if self.sequence_prep == "pad":
            acoustic_data = nn.utils.rnn.pad_sequence(acoustic_data)
            ordered_words = nn.utils.rnn.pad_sequence(ordered_words)

            # swap axes to get (total_inputs, length_of_sequence, length_of_vector)
            acoustic_data = acoustic_data.transpose(0, 1)
            ordered_words = ordered_words.transpose(0, 1)

        elif self.sequence_prep == "truncate":
            if self.truncate_from == "start":
                acoustic_data = [item[-smallest:] for item in acoustic_data]
                ordered_words = [item[-smallest:] for item in ordered_words]
            else:
                acoustic_data = [item[:smallest] for item in acoustic_data]
                ordered_words = [item[:smallest] for item in ordered_words]

            acoustic_data = torch.tensor(acoustic_data)
            ordered_words = torch.tensor(ordered_words)

        print("Acoustic data size is: " + str(acoustic_data.shape))
        print("Ordered words is: " + str(ordered_words.shape))
        print("Ordered speakers size is: " + str(len(ordered_speakers)))
        print("Utterance lengths size is: " + str(len(utt_lengths)))

        print("Data prep and normalization complete")

        # return acoustic info, words indices, speaker
        return acoustic_data, ordered_words, ordered_speakers, utt_lengths

    def combine_acoustic_and_glove(self):
        """
        Combine acoustic feats + glove indices (speaker info, too)
        :return: list of double of ([acoustic], [glove_index], [speaker])
        """
        print("Data prep and normalization starting")
        # print(self.acoustic_dict.keys())
        # print(self.valid_files)
        # sys.exit()

        # set holders for acoustic, words, and speakers
        acoustic_data = []
        ordered_words = []
        ordered_speakers = []

        utt_lengths = []

        # skip the the first cols_to_skip columns
        start_idx = self.cols_to_skip

        # counter for smallest dataframe for truncation
        # get skipped files based on number of items
        #   e.g. too few utts
        if self.sequence_prep == "truncate":
            smallest = self.truncate_seq()

        # get the longest utterance
        longest_utt = get_longest_utterance_asist(
            [
                item
                for key, item in self.acoustic_dict.items()
                if key in self.valid_files
            ]
        )

        # iterate through items in the acoustic dict
        for key, item in self.acoustic_dict.items():
            # if the item has gold data
            if key in self.valid_files:
                speaker_set = set(item["speaker"])
                all_speakers = sorted([str(item) for item in speaker_set])
                # print(all_speakers)
                # sys.exit()

                # prepare intermediate holders
                intermediate_wds = []
                intermediate_speakers = []
                intermediate_acoustic = []

                intermediate_utt_lengths = []

                # for each row in that item's dataframe
                for idx, row in item.iterrows():
                    # get the speaker
                    spkr = str(row["speaker"])
                    # print(row)
                    # sys.exit()

                    # todo: this also includes all researchers
                    #   should we remove them later?
                    intermediate_speakers.append(all_speakers.index(spkr))
                    # print(spkr)
                    # print(all_speakers.index(spkr))
                    # sys.exit()

                    # get the word
                    utt = clean_up_word(row["utt"]).lower()
                    utt_wds = [0] * longest_utt
                    wds = [wd for wd in utt.strip().split(" ")]
                    intermediate_utt_lengths.append(len(wds))
                    # wds = [clean_up_word(wd) for wd in utt.strip().split(" ")]
                    for i, wd in enumerate(wds):
                        # save that word's index
                        if wd in self.glove.wd2idx.keys():
                            utt_wds[i] = self.glove.wd2idx[wd]
                            # utt_wds.append(self.glove.wd2idx[wd])
                        else:
                            utt_wds[i] = self.glove.wd2idx["<UNK>"]
                            # utt_wds.append(self.glove.wd2idx["<UNK>"])

                    intermediate_wds.append(utt_wds)
                    # save the acoustic information in remaining columns
                    row_vals = row.values[start_idx:].tolist()

                    # if using min-max scaling, scale the data
                    if self.norm == "minmax":
                        self.minmax_scale(row_vals, lower=0, upper=1)
                    # add acoustic information to intermediate holder
                    intermediate_acoustic.append(row_vals)
                    # print(row_vals)
                    # sys.exit()

                # add information from intermediate holder to lists of all data
                if self.sequence_prep == "truncate":
                    acoustic_data.append(intermediate_acoustic)
                    ordered_words.append(intermediate_wds)
                    ordered_speakers.append(intermediate_speakers)
                    utt_lengths.append(intermediate_utt_lengths)
                else:
                    acoustic_data.append(torch.tensor(intermediate_acoustic))
                    ordered_words.append(torch.tensor(intermediate_wds))
                    ordered_speakers.append(torch.tensor(intermediate_speakers))
                    utt_lengths.append(torch.tensor(intermediate_utt_lengths))

        # use zero-padding to make all sequences the same length
        # if we need to pad, we MUST pack
        if self.sequence_prep == "pad":
            acoustic_data = nn.utils.rnn.pad_sequence(acoustic_data)
            ordered_words = nn.utils.rnn.pad_sequence(ordered_words)
            ordered_speakers = nn.utils.rnn.pad_sequence(ordered_speakers)
            utt_lengths = nn.utils.rnn.pad_sequence(utt_lengths)

            # swap axes to get (total_inputs, length_of_sequence, length_of_vector)
            acoustic_data = acoustic_data.transpose(0, 1)
            ordered_words = ordered_words.transpose(0, 1)
            ordered_speakers = ordered_speakers.transpose(0, 1)
            utt_lengths = utt_lengths.transpose(0, 1)

        elif self.sequence_prep == "truncate":
            if self.truncate_from == "start":
                acoustic_data = [item[-smallest:] for item in acoustic_data]
                ordered_words = [item[-smallest:] for item in ordered_words]
                ordered_speakers = [item[-smallest:] for item in ordered_speakers]
            else:
                acoustic_data = [item[:smallest] for item in acoustic_data]
                ordered_words = [item[:smallest] for item in ordered_words]
                ordered_speakers = [item[:smallest] for item in ordered_speakers]

            acoustic_data = torch.tensor(acoustic_data)
            ordered_words = torch.tensor(ordered_words)
            ordered_speakers = torch.tensor(ordered_speakers)
            utt_lengths = torch.tensor(utt_lengths)

        print("Acoustic data size is: " + str(acoustic_data.shape))
        print("Ordered words is: " + str(ordered_words.shape))
        print("Ordered speakers size is: " + str(ordered_speakers.shape))
        print("Utterance lengths size is: " + str(utt_lengths.shape))

        print("Data prep and normalization complete")

        # return acoustic info, words indices, speaker
        return acoustic_data, ordered_words, ordered_speakers, utt_lengths

    def create_ordered_ys(self):
        """
        create a list of all outcomes in the same order as the data
        """
        # create holder
        ordered_ys = []

        # set index for ys dataframe to sid to use it in search
        ys = self.ys_df.set_index(["sid"])

        # for each key in acoustic dict's keys
        for key in self.acoustic_dict.keys():
            # if the sid has gold data, add it
            if key in self.valid_files:
                print(key)
                if key not in self.skipped_files:
                    y = ys[ys.index == key].overall.item()
                    ordered_ys.append(y)

        # return ordered list
        return ordered_ys

    def create_ordered_ys_utt_level(self, num_utts):
        """
        create a list of all outcomes in the same order as the data
        """
        # create holder
        # ordered_ys = random.sample(range(1), num_utts)
        ordered_ys = [random.randint(0, 1) for _ in range(num_utts)]
        return ordered_ys

    def minmax_scale(self, df, lower=0, upper=1):
        # perform min-max scaling
        for i, val in enumerate(df):
            new_val = scale_feature(
                val,
                self.min_max_scaler.min(i),
                self.min_max_scaler.max(i),
                lower,
                upper,
            )
            df[i] = new_val
        return df

    def truncate_seq(self, starting_size=1e10, minimum=1000):
        """
        truncate the sequence of data
        :return smallest_item: the size of the smallest item
        """
        smallest_item = starting_size
        skipped_files = []
        for key, item in self.acoustic_dict.items():
            size = item.shape[0]
            if size < smallest_item:
                if size >= minimum:
                    smallest_item = size
                else:
                    skipped_files.append(key)
        self.skipped_files = skipped_files

        return smallest_item


def save_asist_data(
    dataset,
    save_path,
    data_path,
    feature_set,
    glove_path,
    emb_type,
    feats_to_use=None,
    data_as_dict=False,
    avg_acoustic_data=False,
    custom_feats_file=None,
    include_spectrograms=False,
    num_partitions=1
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

    train_ds, dev_ds, test_ds, clss_weights = prep_asist_data(
        data_path,
        feature_set,
        emb_type,
        glove_path,
        feats_to_use,
        data_as_dict,
        avg_acoustic_data,
        custom_feats_file,
        include_spectrograms
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

    if num_partitions == 3:
        pickle.dump(train_ds,open(f"{train_save_name}.pickle", "wb"))
        pickle.dump(dev_ds, open(f"{dev_save_name}.pickle", "wb"))
        pickle.dump(test_ds, open(f"{test_save_name}.pickle", "wb"))
        pickle.dump(clss_weights, open(f"{wts_save_name}.pickle", "wb"))
    elif num_partitions == 1:
        data = train_ds + dev_ds + test_ds
        pickle.dump(data, open(f"{test_save_name}.pickle", 'wb'))


if __name__ == "__main__":
    save_path = "../../dataset/pickled_data"
    # data_path = "../../PROJECTS/ToMCAT/Evaluating_modelpredictions/data_from_speechAnalyzer/used_for_evaluating_model_results"
    data_path = "../../study3_data_to_annotate"
    emb_type = "glove"
    glove_path = "../../datasets/glove/glove.subset.300d.txt"
    dict_data = True
    avg_feats = True
    with_spec = False
    partitions = 1

    save_asist_data("asist",
                    save_path=save_path,
                    data_path=data_path,
                    feature_set="IS13",
                    glove_path=glove_path,
                    emb_type=emb_type,
                    data_as_dict=dict_data,
                    avg_acoustic_data=avg_feats,
                    include_spectrograms=with_spec,
                    num_partitions=partitions)
