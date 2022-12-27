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

random.seed(345)


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

    def __init__(
        self,
        train_data,
        dev_data,
        test_data,
        class_loss_func,
        task_num,
        binary=False,
        optimizer=None,
    ):
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

    def __init__(
        self, test_data, class_loss_func, task_num, binary=False, optimizer=None
    ):
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


# todo: test out using this sampler
class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    Slightly altered from: https://gist.github.com/bomri/d93da3e6f840bb93406f40a6590b9c48
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max(
            [len(cur_dataset.samples) for cur_dataset in dataset.datasets]
        )

    def __len__(self):
        return (
            self.batch_size
            * math.ceil(self.largest_dataset_size / self.batch_size)
            * len(self.dataset.datasets)
        )

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # stop trying to add samples and continue on in the next dataset
                        break
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        # sampler_iterators[i] = samplers_list[i].__iter__()
                        # cur_batch_sampler = sampler_iterators[i]
                        # cur_sample_org = cur_batch_sampler.__next__()
                        # cur_sample = cur_sample_org + push_index_val[i]
                        # cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


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
    word = word.replace(".", " . ")
    word = word.replace(",", " , ")
    word = word.replace("!", " ! ")
    word = word.replace("?", " ? ")
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


def create_data_folds_list(data, perc_train, perc_test, shuffle=True):
    """
    Create train, dev, and test data folds
    Specify the percentage that goes into each
    data: A LIST of the data that goes into all folds
    perc_* : the percentage for each fold
    Percentage not included in train or test fold allocated to dev
    shuffle: whether to shuffle the data
    """
    if shuffle:
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


def get_data_samples(data_list, num_samples):
    # get a new set of training data samples
    # given data and number of samples
    if num_samples > len(data_list):
        new_samples = random.choices(data_list, num_samples)
    else:
        new_samples = random.sample(data_list, num_samples)

    return new_samples


def get_class_weights(data_tensors_dict, data_type):
    if data_type == "meld":
        y_tensor = data_tensors_dict["all_emotions"]
    elif data_type == "mustard":
        y_tensor = data_tensors_dict["all_sarcasm"]
    elif data_type == "chalearn" or data_type == "firstimpr":
        ys = [
            [
                data_tensors_dict["all_extraversion"][i],
                data_tensors_dict["all_neuroticism"][i],
                data_tensors_dict["all_agreeableness"][i],
                data_tensors_dict["all_openness"][i],
                data_tensors_dict["all_conscientiousness"][i],
            ]
            for i, item in enumerate(data_tensors_dict["all_extraversion"])
        ]
        y_tensor = [item.index(max(item)) for item in ys]
    # if ravdess, this is fed the processed list and not a tensor dict
    elif data_type == "ravdess":
        y_tensor = data_tensors_dict
    else:
        print(
            "data type does not have associated ys to get class weights; "
            "if this dataset is not prepartitioned, weights will be assigned later."
        )
        return None


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

    return mean, stdev


def get_acoustic_means(acoustic_data):
    """
    Get averages and standard deviations of acoustic data
    Should deal with 2-d and 3-d tensors
    param acoustic_data : the acoustic data in tensor format
    """
    if len(acoustic_data.shape) == 3:
        # reshape data + calculate means
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


def get_nonzero_avg(tensor):
    """
    Get the average of all non-padding vectors in a tensor
    """
    nonzeros = tensor.sum(axis=1) != 0
    num_nonzeros = sum(i == True for i in nonzeros)

    nonzero_avg = tensor.sum(axis=0) / num_nonzeros

    return nonzero_avg


def look_up_num_classes(dataset_name):
    """
    Look up the number of classes in a dataset if given the name
    :param dataset_name: Name of the dataset of interest
    :return:
    """
    # todo: need to fix this to be name of classes instead of number
    #   because of 2 sets of emotion tags
    all_datasets = {
        "cdc": 2,
        "mosi_ternary": 3,
        "mosi_all": 7,
        "firstimpr_binary": 2,
        "firstimpr_ternary": 3,
        "firstimpr_maxclass": 5,
        "meld": [7, 3],
        "mustard": 2,
        "ravdess": [8, 2],
    }

    return all_datasets[dataset_name]


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


def split_string_time(timestamp):
    """
    split "hh:mm:ss.sss" timestamps to seconds + ms
    used to calculate start and end of acoustic features
    """
    time = timestamp.split(":")
    if len(time) == 3:
        return (float(time[0]) * 60 + float(time[1])) * 60 + float(time[2])
    elif len(time) == 2:
        return float(time[0]) * 60 + float(time[1])


def transform_acoustic_item(item, acoustic_means, acoustic_stdev):
    """
    Use gender averages and stdev to transform an acoustic item
    item : a 1D tensor
    acoustic_means : the appropriate vector of means
    acoustic_stdev : the corresponding stdev vector
    """
    return (item - acoustic_means) / acoustic_stdev
