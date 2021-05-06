# prepare text and audio for use in neural network models
import math
import os
import random
import sys
from collections import OrderedDict

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertModel

import statistics


# classes
from torch.utils.data.sampler import RandomSampler


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


class MultitaskObject(object):
    """
    An object to hold the data and meta-information for each of the datasets/tasks
    """
    def __init__(self, train_data, dev_data, test_data, class_loss_func, task_num, binary=False,
                 optimizer=None):
        """
        train_data, dev_data, and test_data are DatumListDataset datasets
        """
        self.train = train_data
        self.dev = dev_data
        self.test = test_data
        self.loss_fx = class_loss_func
        self.optimizer = optimizer
        self.task_num = task_num
        self.binary = binary
        self.loss_multiplier = 1

    def change_loss_multiplier(self, multiplier):
        """
        Add a different loss multiplier to task
        This will be used as a multiplier for loss in multitask network
        e.g. if weight == 1.5, loss = loss * 1.5
        """
        self.loss_multiplier = multiplier


class MultitaskTestObject(object):
    """
    An object to hold the data and meta-information for each of the datasets/tasks
    """
    def __init__(self, test_data, class_loss_func, task_num, binary=False,
                 optimizer=None):
        """
        train_data, dev_data, and test_data are DatumListDataset datasets
        """
        self.test = test_data
        self.loss_fx = class_loss_func
        self.optimizer = optimizer
        self.task_num = task_num
        self.binary = binary
        self.loss_multiplier = 1

    def change_loss_multiplier(self, multiplier):
        """
        Add a different loss multiplier to task
        This will be used as a multiplier for loss in multitask network
        e.g. if weight == 1.5, loss = loss * 1.5
        """
        self.loss_multiplier = multiplier

# todo: will we need this?
# class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
#     """
#     iterate over tasks and provide a random batch per task in each mini-batch
#     Slightly altered from: https://gist.github.com/bomri/d93da3e6f840bb93406f40a6590b9c48
#     """
#     def __init__(self, dataset, batch_size):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.number_of_datasets = len(dataset.datasets)
#         self.largest_dataset_size = max([len(cur_dataset.samples) for cur_dataset in dataset.datasets])
#
#     def __len__(self):
#         return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)
#
#     def __iter__(self):
#         samplers_list = []
#         sampler_iterators = []
#         for dataset_idx in range(self.number_of_datasets):
#             cur_dataset = self.dataset.datasets[dataset_idx]
#             sampler = RandomSampler(cur_dataset)
#             samplers_list.append(sampler)
#             cur_sampler_iterator = sampler.__iter__()
#             sampler_iterators.append(cur_sampler_iterator)
#
#         push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
#         step = self.batch_size * self.number_of_datasets
#         samples_to_grab = self.batch_size
#         # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
#         epoch_samples = self.largest_dataset_size * self.number_of_datasets
#
#         final_samples_list = []  # this is a list of indexes from the combined dataset
#         for _ in range(0, epoch_samples, step):
#             for i in range(self.number_of_datasets):
#                 cur_batch_sampler = sampler_iterators[i]
#                 cur_samples = []
#                 for _ in range(samples_to_grab):
#                     try:
#                         cur_sample_org = cur_batch_sampler.__next__()
#                         cur_sample = cur_sample_org + push_index_val[i]
#                         cur_samples.append(cur_sample)
#                     except StopIteration:
#                         # stop trying to add samples and continue on in the next dataset
#                         break
#                         # got to the end of iterator - restart the iterator and continue to get samples
#                         # until reaching "epoch_samples"
#                         # sampler_iterators[i] = samplers_list[i].__iter__()
#                         # cur_batch_sampler = sampler_iterators[i]
#                         # cur_sample_org = cur_batch_sampler.__next__()
#                         # cur_sample = cur_sample_org + push_index_val[i]
#                         # cur_samples.append(cur_sample)
#                 final_samples_list.extend(cur_samples)
#
#         return iter(final_samples_list)


class Glove(object):
    def __init__(self, glove_dict):
        """
        Use a dict of format {word: vec} to get torch.tensor of vecs
        :param glove_dict: a dict created with make_glove_dict
        """
        self.glove_dict = OrderedDict(glove_dict)
        self.data = self.create_embedding()
        self.wd2idx = self.get_index_dict()
        self.idx2glove = self.get_index2glove_dict()  # todo: get rid of me
        self.max_idx = -1

        # add an average <UNK> if not in glove dict
        if "<UNK>" not in self.glove_dict.keys():
            mean_vec = self.get_avg_embedding()
            self.add_vector("<UNK>", mean_vec)

    def id_or_unk(self, t):
        if t.strip() in self.wd2idx:
            return self.wd2idx[t.strip()]
        else:
            # print(f"OOV: [[{t}]]")
            return self.wd2idx["<UNK>"]

    def index(self, toks):
        return [self.id_or_unk(t) for t in toks]

    def index_with_counter(self, toks, counter):
        for tok in toks:
            if tok in counter.keys() and counter[tok] > 4:
                return [self.id_or_unk(tok)]
            else:
                return [self.wd2idx["<UNK>"]]

    def create_embedding(self):
        emb = []
        for vec in self.glove_dict.values():
            emb.append(vec)
        return torch.tensor(emb)

    def get_embedding_from_index(self, idx):
        return self.idx2glove[idx]

    def get_index_dict(self):
        # create word: index dict
        c = 0
        wd2idx = {}
        for k in self.glove_dict.keys():
            wd2idx[k] = c
            c += 1
        self.max_idx = c
        return wd2idx

    def get_index2glove_dict(self):
        # create index: vector dict
        c = 0
        idx2glove = {}
        for k, v in self.glove_dict.items():
            idx2glove[self.wd2idx[k]] = v
        return idx2glove

    def add_vector(self, word, vec):
        # adds a new word vector to the dictionaries
        self.max_idx += 1
        if self.max_idx not in self.wd2idx.keys():
            self.glove_dict[word] = vec  # add to the glove dict
            self.wd2idx[word] = self.max_idx  # add to the wd2idx dict
            self.idx2glove[self.max_idx] = vec  # add to the idx2 glove dict
            torch.cat(
                (self.data, vec.unsqueeze(dim=0)), dim=0
            )  # add to the data tensor

    def get_avg_embedding(self):
        # get an average of all embeddings in dataset
        # can be used for "<UNK>" if it doesn't exist
        return torch.mean(self.data, dim=0)


class MinMaxScaleRange:
    """
    A class to calculate mins and maxes for each feature in the data in order to
    use min-max scaling
    """

    def __init__(self,):
        self.mins = {}
        self.maxes = {}

    def update(self, key, val):
        if (
            key in self.mins.keys() and val < self.mins[key]
        ) or key not in self.mins.keys():
            self.mins[key] = val
        if (
            key in self.maxes.keys() and val > self.maxes[key]
        ) or key not in self.maxes.keys():
            self.maxes[key] = val

    def contains(self, key):
        if key in self.mins.keys():
            return True
        else:
            return False

    def min(self, key):
        try:
            return self.mins[key]
        except KeyError:
            return "The key {0} does not exist in mins".format(key)

    def max(self, key):
        try:
            return self.maxes[key]
        except KeyError:
            return "The key {0} does not exist in maxes".format(key)


# helper functions


def clean_up_word(word):
    word = word.replace("\x92", "'")
    word = word.replace("\x91", " ")
    word = word.replace("\x97", "-")
    word = word.replace("\x93", " ")
    word = word.replace("[", " ")
    word = word.replace("]", " ")
    word = word.replace("-", " - ")
    word = word.replace("%", " % ")
    word = word.replace("@", " @ ")
    word = word.replace("$", " $ ")
    word = word.replace("...", " ... ")
    word = word.replace("/", " / ")
    # clean up word by putting in lowercase + removing punct
    # punct = [
    #     ",",
    #     ".",
    #     "!",
    #     "?",
    #     ";",
    #     ":",
    #     "'",
    #     '"',
    #     "-",
    #     "$",
    #     "’",
    #     "…",
    #     "[",
    #     "]",
    #     "(",
    #     ")",
    # ]
    # for char in word:
    #     if char in punct:
    #         word = word.replace(char, " ")
    if word.strip() == "":
        word = "<UNK>"
    return word


def create_data_folds(data, perc_train, perc_test):
    """
    Create train, dev, and test folds for a dataset without them
    Specify the percentage of the data that goes into each fold
    data : a Pandas dataframe with (at a minimum) gold labels for all data
    perc_* : the percentage for each fold
    Percentage not included in train or test fold allocated to dev
    """
    # shuffle the rows of the dataframe
    shuffled = data.sample(frac=1).reset_index(drop=True)

    # get length of df
    length = shuffled.shape[0]

    # calculate length of each split
    train_len = perc_train * length
    test_len = perc_test * length

    # get slices of dataset
    train_data = shuffled.iloc[: int(train_len)]
    test_data = shuffled.iloc[int(train_len) : int(train_len) + int(test_len)]
    dev_data = shuffled.iloc[int(train_len) + int(test_len) :]

    # return data
    return train_data, dev_data, test_data


def create_data_folds_list(data, perc_train, perc_test):
    """
    Create train, dev, and test data folds
    Specify the percentage that goes into each
    data: A LIST of the data that goes into all folds
    perc_* : the percentage for each fold
    Percentage not included in train or test fold allocated to dev
    """
    # shuffle the data
    random.shuffle(data)

    # get length
    length = len(data)

    # calculate proportion alotted to train and test
    train_len = math.floor(perc_train * length)
    test_len = math.floor(perc_test * length)

    # get datasets
    train_data = data[:train_len]
    test_data = data[train_len : train_len + test_len]
    dev_data = data[train_len + test_len :]

    # return data
    return train_data, dev_data, test_data


def get_avg_vec(nested_list):
    # get the average vector of a nested list
    # used for utterance-level feature averaging
    return [statistics.mean(item) for item in zip(*nested_list)]


# I found this method recently, in a discussion that sometimes weights are better
# served in the loss function than in a sampler.  What you were returning below
# seem to be counts, not weights.  These are automatically calculated by sklearn, and
# apparently based off imbalanced logistic regression.  Let's see if they help!
def get_class_weights(y_tensor):
    labels = [int(y) for y in y_tensor]
    classes = sorted(list(set(labels)))
    weights = compute_class_weight("balanced", classes, labels)
    return torch.tensor(weights, dtype=torch.float)


# def get_class_weights(y_set):
#     class_counts = {}
#     y_values = y_set.tolist()

#     num_labels = max(y_values) + 1

#     for item in y_values:
#         if item not in class_counts:
#             class_counts[item] = 1
#         else:
#             class_counts[item] += 1
#     class_weights = [0.0] * num_labels
#     for k, v in class_counts.items():
#         class_weights[k] = float(v)
#     class_weights = torch.tensor(class_weights)
#     return class_weights


def get_gender_avgs(acoustic_data, gender_set, gender=1):
    """
    Get averages and standard deviations split by gender
    param acoustic_data : the acoustic data (tensor format)
    param gender : the gender to return avgs for; 0 = all, 1 = f, 2 = m
    """
    all_items = []

    for i, item in enumerate(acoustic_data):
        if gender_set[i] == gender:
            all_items.append(torch.tensor(item))

    all_items = torch.stack(all_items)

    mean, stdev = get_acoustic_means(all_items)
    # mean = all_items.mean(dim=0, keepdim=False)
    # stdev = all_items.std(dim=0, keepdim=False)

    return mean, stdev


def get_acoustic_means(acoustic_data):
    """
    Get averages and standard deviations of acoustic data
    Should deal with 2-d and 3-d tensors
    param acoustic_data : the acoustic data in tensor format
    """
    # print("Now starting to calculate acoustic means")
    # print(acoustic_data.shape)
    if len(acoustic_data.shape) == 3:
        # reshape data + calculate means
        # can do data.mean(dim=0).mean(dim=0), BUT runs out of memory
        dim_0 = acoustic_data.shape[0]
        dim_1 = acoustic_data.shape[1]
        dim_2 = acoustic_data.shape[2]
        reshaped_data = torch.reshape(acoustic_data, (dim_0 * dim_1, dim_2))
        mean = reshaped_data.mean(dim=0, keepdim=False)
        stdev = reshaped_data.std(dim=0, keepdim=False)
    elif len(acoustic_data.shape) == 2:
        mean = acoustic_data.mean(dim=0, keepdim=False)
        stdev = acoustic_data.std(dim=0, keepdim=False)

    return mean, stdev


def get_longest_utterance(pd_dataframes):
    """
    Get the longest utterance in the dataset
    :param pd_dataframes: the dataframes for the dataset
    :return:
    """
    max_length = 0
    for item in pd_dataframes:
        for i in range(item["utt_num"].max()):
            utterance = item.loc[item["utt_num"] == i + 1]
            utt_length = utterance.shape[0]
            if utt_length > max_length:
                max_length = utt_length
    return max_length


def get_longest_utt(utts_list):
    """
    checks lengths of utterances in a list
    and return len of longest
    """
    longest = 0

    for utt in utts_list:
        try:
            split_utt = utt.strip().split(" ")
            utt_len = len(split_utt)
            if utt_len > longest:
                longest = utt_len
        except AttributeError:
            # if utterance is empty may be read as float
            continue

    return longest


def get_max_num_acoustic_frames(acoustic_set):
    """
    Get the maximum number of acoustic feature frames in any utterance
    from the dataset used
    acoustic_set : the FULL set of acoustic dfs (train + dev + test)
    """
    longest = 0

    for feats_df in acoustic_set:
        utt_len = len(feats_df)
        # utt_len = feats_df.shape[0]
        if utt_len > longest:
            longest = utt_len

    return longest


def get_speaker_gender(idx2gender_path):
    """
    Get the gender of each speaker in the list
    Includes 0 as UNK, 1 == F, 2 == M
    """
    speaker_df = pd.read_csv(idx2gender_path, usecols=["idx", "gender"])

    return dict(zip(speaker_df.idx, speaker_df.gender))


def get_speaker_to_index_dict(speaker_set):
    """
    Take a set of speakers and return a speaker2idx dict
    speaker_set : the set of speakers
    """
    # set counter
    speaker_num = 0

    # set speaker2idx dict
    speaker2idx = {}

    # put all speakers in
    for speaker in speaker_set:
        speaker2idx[speaker] = speaker_num
        speaker_num += 1

    return speaker2idx


def make_acoustic_dict(
    acoustic_path,
    f_end="_IS09_avgd.csv",
    use_cols=None,
    data_type="clinical",
    files_to_get=None,
):
    """
    makes a dict of (sid, call): data for use in ClinicalDataset objects
    f_end: end of acoustic file names
    use_cols: if set, should be a list [] of column names to include
    """
    acoustic_dict = {}
    for f in os.listdir(acoustic_path):
        if f.endswith(f_end):
            if files_to_get is None or "_".join(f.split("_")[:2]) in files_to_get:
                if use_cols is not None:
                    try:
                        feats = pd.read_csv(acoustic_path + "/" + f, usecols=use_cols)
                    except ValueError:
                        # todo: add warning
                        feats = []
                else:
                    feats = pd.read_csv(acoustic_path + "/" + f)
                if data_type == "asist":
                    label = f.split("_")
                    if label[1] == "mission":
                        sid = label[0]
                        mission_id = label[2]
                    else:
                        try:
                            sid = int(label[1])
                        except ValueError:
                            sid = int(label[1].split("-")[1])
                        mission_id = 0  # later iterations of this should have mission IDs
                    acoustic_dict[(sid, mission_id)] = feats
                    # callid = f.split("_")[2]  # asist data has format sid_mission_num
                else:
                    sid = f.split("_")[0]
                    # clinical data has format sid_callid
                    # meld has format dia_utt
                    callid = f.split("_")[1]
                    acoustic_dict[(sid, callid)] = feats
    return acoustic_dict


def make_acoustic_set(
    text_path,
    acoustic_dict,
    data_type,
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
        all_utts_df = pd.read_csv(text_path)
    elif type(text_path) == pd.core.frame.DataFrame:
        all_utts_df = text_path
    else:
        sys.exit("text_path is of unaccepted type.")

    # get lists of valid dialogues and utterances
    if data_type == "meld":
        valid_dia_utt = all_utts_df["DiaID_UttID"].tolist()
    else:
        valid_dia_utt = all_utts_df["clip_id"].tolist()

    # set holders for acoustic data
    all_acoustic = []
    usable_utts = []

    # for all items with audio + gold label
    for idx, item in enumerate(valid_dia_utt):
        # print(idx, item)
        # if that dialogue and utterance appears has an acoustic feats file
        if (item.split("_")[0], item.split("_")[1]) in acoustic_dict.keys():
            # print(f"{item} was found")
            # pull out the acoustic feats dataframe
            acoustic_data = acoustic_dict[(item.split("_")[0], item.split("_")[1])]

            # add this dialogue + utt combo to the list of possible ones
            usable_utts.append((item.split("_")[0], item.split("_")[1]))

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
                    # acoustic_holder = torch.mean(torch.tensor(acoustic_data), dim=0)
                    # acoustic_holder = torch.mean(torch.tensor(acoustic_data)[10:min(1491, len(acoustic_data) - 9)], dim=0)
                    # try skipping first and last 5%
                    # acoustic_holder = torch.mean(torch.tensor(acoustic_data)[math.floor(data_len * 0.05):math.ceil(data_len * 0.95)], dim=0)
                    # try skipping first and last 25% 15%
                    data_len = len(acoustic_data)
                    # acoustic_holder = torch.rand(76 * 3)
                    # try just getting within the certain range of frames
                    # acoustic_mean = torch.mean(torch.tensor(acoustic_data), dim=0)
                    # acoustic_holder = torch.mean(torch.tensor(acoustic_data)[math.floor(data_len * 0.25):math.ceil(data_len * 0.75)], dim=0)
                    # acoustic_max = torch.max(torch.tensor(acoustic_data), dim=0).values
                    # acoustic_min = torch.min(torch.tensor(acoustic_data), dim=0).values
                    # acoustic_stdev = torch.std(torch.tensor(acoustic_data), dim=0)

                    acoustic_mean = torch.mean(torch.tensor(acoustic_data)[math.floor(data_len * 0.25):math.ceil(data_len * 0.75)], dim=0)
                    acoustic_max = torch.max(torch.tensor(acoustic_data)[math.floor(data_len * 0.25):math.ceil(data_len * 0.75)], dim=0).values
                    acoustic_min = torch.min(torch.tensor(acoustic_data)[math.floor(data_len * 0.25):math.ceil(data_len * 0.75)], dim=0).values
                    # acoustic_med = torch.median(torch.tensor(acoustic_data)[math.floor(data_len * 0.25):math.ceil(data_len * 0.75)], dim=0)[0]
                    acoustic_stdev = torch.std(torch.tensor(acoustic_data)[math.floor(data_len * 0.25):math.ceil(data_len * 0.75)], dim=0)

                    acoustic_meanplus = acoustic_mean + acoustic_stdev
                    acoustic_meanminus = acoustic_mean - acoustic_stdev
                    acoustic_holder = torch.cat((acoustic_mean, acoustic_max, acoustic_min, acoustic_meanplus, acoustic_meanminus), dim=0)
                    # acoustic_holder = torch.cat((acoustic_mean, acoustic_meanplus, acoustic_meanminus), dim=0)
                    # acoustic_holder = torch.cat((acoustic_mean, acoustic_max, acoustic_min), dim=0)

                    # x = torch.tensor(acoustic_data)
                    # print(acoustic_mean)
                    # print(torch.mean(torch.tensor(acoustic_data)[:math.ceil(data_len * 0.75)], dim=0))
                    # torch.mean(torch.tensor(acoustic_data)[math.floor(data_len * 0.25):math.ceil(data_len * 0.5)],
                    #            dim=0)
                    # print(x.shape)
                    # print(x[5:25].shape)
                    # print(x[:25].shape)
                    # print(x[math.floor(data_len * 0.25):math.ceil(data_len * 0.75)].shape)
                    # print(x[math.floor(data_len * 0.25):math.ceil(data_len * 0.5)].shape)
                    # y = x[math.floor(data_len * 0.25):math.ceil(data_len * 0.5)].shape
                    # print(torch.m)
                    # print(x[:math.ceil(data_len * 0.75)].shape)
                    # print(len(acoustic_data[:math.ceil(data_len * 0.75)]))
                    # exit()



                    # print(acoustic_holder.shape)
                    # acoustic_holder = torch.cat((acoustic_means, acoustic_meanplus, acoustic_meanminus))
                    # acoustic_holder = torch.cat((acoustic_means, acoustic_med, acoustic_stdev), 0)
                    # get average of all non-padding vectors
                    # nonzero_avg = get_nonzero_avg(torch.tensor(acoustic_data))
                    # acoustic_holder = nonzero_avg

            # add features as tensor to acoustic data
            all_acoustic.append(acoustic_holder)

    # pad the sequence and reshape it to proper format
    # this is here to keep the formatting for acoustic RNN
    all_acoustic = nn.utils.rnn.pad_sequence(all_acoustic)
    all_acoustic = all_acoustic.transpose(0, 1)

    return all_acoustic, usable_utts


def get_nonzero_avg(tensor):
    """
    Get the average of all non-padding vectors in a tensor
    """
    nonzeros = tensor.sum(axis=1) != 0
    num_nonzeros = sum(i == True for i in nonzeros)

    nonzero_avg = tensor.sum(axis=0) / num_nonzeros

    return nonzero_avg


def make_glove_dict(glove_path):
    """creates a dict of word: embedding pairs
    :param glove_path: the path to our glove file
    (includes name of file and extension)
    """
    glove_dict = {}
    with open(glove_path) as glove_file:
        for line in glove_file:
            line = line.rstrip().split(" ")
            glove_dict[line[0]] = [float(item) for item in line[1:]]
    return glove_dict


def perform_feature_selection(xs, ys, num_to_keep):
    """
    Perform feature selection on the dataset
    """
    new_xs = SelectKBest(chi2, k=num_to_keep).fit_transform(xs, ys)
    return new_xs


def scale_feature(value, min_val, max_val, lower=0.0, upper=1.0):
    # scales a single feature using min-max normalization
    if min_val == max_val:
        return upper
    else:
        # the result will be a value in [lower, upper]
        return lower + (upper - lower) * (value - min_val) / (max_val - min_val)


def transform_acoustic_item(item, acoustic_means, acoustic_stdev):
    """
    Use gender averages and stdev to transform an acoustic item
    item : a 1D tensor
    acoustic_means : the appropriate vector of means
    acoustic_stdev : the corresponding stdev vector
    """
    return (item - acoustic_means) / acoustic_stdev