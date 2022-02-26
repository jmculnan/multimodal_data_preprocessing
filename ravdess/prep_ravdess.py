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
    get_data_samples,
)


def prep_ravdess_data(
    data_path="../../datasets/multimodal_datasets/RAVDESS_Speech",
    feature_set="IS13",
    embedding_type="distilbert",
    glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
    features_to_use=None,
    as_dict=False,
    avg_acoustic_data=False,
    custom_feats_file=None,
    selected_ids=None,
    num_train_ex=None,
    include_spectrograms=False,
):
    # load glove
    if embedding_type.lower() == "glove":
        glove_dict = make_glove_dict(glove_filepath)
        glove = Glove(glove_dict)
    else:
        glove = None

    # create instance of StandardPrep class
    ravdess_prep = RavdessPrep(
        ravdess_path=data_path,
        feature_set=feature_set,
        glove=glove,
        train_prop=0.6,
        test_prop=0.2,
        f_end=f"{feature_set}.csv",
        use_cols=features_to_use,
        as_dict=as_dict,
        avg_acoustic_data=avg_acoustic_data,
        custom_feats_file=custom_feats_file,
        selected_ids=selected_ids,
        embedding_type=embedding_type,
        include_spectrograms=include_spectrograms
    )

    train_data = ravdess_prep.train_data
    dev_data = ravdess_prep.dev_data
    test_data = ravdess_prep.test_data

    # get class weights
    # todo: allow to get emotion or intensity or both
    class_weights = ravdess_prep.intensity_weights

    if num_train_ex:
        train_data = get_data_samples(train_data, num_train_ex)

    return train_data, dev_data, test_data, class_weights


class RavdessPrep:
    """
    A class to prepare ravdess data
    """

    def __init__(
        self,
        ravdess_path,
        feature_set,
        glove,
        train_prop=0.6,
        test_prop=0.2,
        f_end="IS10.csv",
        use_cols=None,
        as_dict=False,
        avg_acoustic_data=False,
        custom_feats_file=None,
        selected_ids=None,
        embedding_type="distilbert",
        include_spectrograms=False,  # todo: still need to add this
    ):
        # path to dataset--all within acoustic files for ravdess
        self.path = ravdess_path

        # get path to acoustic feature location
        if custom_feats_file is None:
            self.feature_path = f"{ravdess_path}/{feature_set}"
        else:
            self.feature_path = ravdess_path

        # get tokenizer
        self.tokenizer = get_tokenizer("basic_english")

        # get data tensors
        if custom_feats_file is None:
            self.all_data = make_ravdess_data_tensors(
                self.feature_path,
                glove,
                f_end,
                use_cols,
                add_avging=avg_acoustic_data,
                as_dict=as_dict,
                bert_type=embedding_type,
            )
        else:
            self.all_data = make_ravdess_data_tensors_with_custom_acoustic_features(
                self.feature_path,
                custom_feats_file,
                glove,
                use_cols,
                as_dict=as_dict,
                selected_ids=selected_ids,
                bert_type=embedding_type,
            )

        # get all used ids
        self.used_ids = self.get_all_used_ids(self.all_data)

        if include_spectrograms:
            self.spec_dict, self.spec_lengths_dict = make_spectrograms_dict(self.path, "ravdess")
            # self.spec_list, self.spec_lengths_list = self.make_spectrogram_set(self.spec_dict, self.spec_lengths_dict)

            self.update_data_tensors()

        if custom_feats_file:
            (self.train_data, self.dev_data, self.test_data,) = create_data_folds_list(
                self.all_data, train_prop, test_prop, shuffle=False
            )
        else:
            (self.train_data, self.dev_data, self.test_data,) = create_data_folds_list(
                self.all_data, train_prop, test_prop
            )

        # pull out ys from train to get class weights
        if as_dict:
            self.train_y_intensity = torch.tensor(
                [item["ys"][0] for item in self.train_data]
            )
            self.train_y_emotion = torch.tensor(
                [item["ys"][1] for item in self.train_data]
            )
        else:
            self.train_y_intensity = torch.tensor([item[4] for item in self.train_data])
            self.train_y_emotion = torch.tensor([item[5] for item in self.train_data])

        # set the sarcasm weights
        self.emotion_weights = get_class_weights(
            self.train_y_emotion, data_type="ravdess"
        )
        self.intensity_weights = get_class_weights(
            self.train_y_intensity, data_type="ravdess"
        )

    def get_all_used_ids(self, all_data):
        """
        Get a list of all the ids that have both acoustic and text/gold info
        :return: array of all valid ids
        """
        all_used_ids = [item['audio_id'] for item in all_data]

        return all_used_ids

    def make_spectrogram_set(
        self,
        spec_dict,
        spec_lengths_dict,
        longest_spec=1500
    ):
        """
        Prepare the spectrogram data
        :param spec_dict: a dict of acoustic feat dfs
        :param spec_lengths_dict: a dict of lengths of spectrograms
        :param longest_spec: the longest allowed spec df
        :return:
        """

        # set holders for acoustic data
        all_spec = []
        ordered_spec_lengths = []

        # for all items with audio + gold label
        for item in tqdm(self.used_ids, desc=f"Preparing spec set for ravdess"):

            # pull out the spec feats df
            spec_data = spec_dict[item]
            ordered_spec_lengths.append(spec_lengths_dict[item])

            spec_data = spec_data[spec_data.index <= longest_spec]
            spec_holder = torch.tensor(spec_data.values)

            # add features as tensor to acoustic data
            all_spec.append(spec_holder)

        # delete acoustic dict to save space
        del spec_dict

        print(f"Acoustic set made at {datetime.datetime.now()}")

        return all_spec, ordered_spec_lengths

    def update_data_tensors(self):
        """
        Add spectrogram info to data tensors if needed
        """
        for item in self.all_data:
            audio_id = item['audio_id']

            item['x_spec'] = self.spec_dict[audio_id]
            item['spec_length'] = self.spec_lengths_dict[audio_id]



def make_ravdess_data_tensors(
    acoustic_path,
    glove=None,
    f_end="_IS10.csv",
    use_cols=None,
    add_avging=True,
    as_dict=False,
    bert_type="distilbert",
):
    """
    makes data tensors for use in RAVDESS objects
    f_end: end of acoustic file names
    use_cols: if set, should be a list [] of column names to include
    n_to_skip : the number of columns at the start to ignore (e.g. name, time)
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
    audio_ids = []

    # holder for all data
    data = []

    if glove is not None:
        utt_1 = glove.index(["kids", "are", "talking", "by", "the", "door"])
        utt_2 = glove.index(["dogs", "are", "sitting", "by", "the", "door"])
        utt_length = 6
    else:
        utt_1_text = "kids are talking by the door"
        utt_2_text = "dogs are sitting by the door"
        # instantiate embeddings maker
        if bert_type.lower() == "bert":
            emb_maker = BertEmb()
            utt_1, id_1 = emb_maker.tokenize(utt_1_text)
            utt_1 = emb_maker.get_embeddings(utt_1, torch.tensor(id_1).unsqueeze(0), 8)
            utt_2, id_2 = emb_maker.tokenize(utt_2_text)
            utt_2 = emb_maker.get_embeddings(utt_2, torch.tensor(id_2).unsqueeze(0), 8)
        else:
            emb_maker = DistilBertEmb()
            utt_1, id_1 = emb_maker.tokenize(utt_1_text)
            utt_1 = emb_maker.get_embeddings(utt_1, torch.tensor(id_1), 8)
            utt_2, id_2 = emb_maker.tokenize(utt_2_text)
            utt_2 = emb_maker.get_embeddings(utt_2, torch.tensor(id_2), 8)
        utt_length = max(len(utt_1), len(utt_2))

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
                audio_ids.append(all_labels)

    # convert data to torch tensors
    if glove is not None:
        utterances = torch.tensor(utterances)
    else:
        utterances = torch.stack(utterances)
        utterances = torch.squeeze(utterances, dim=0)
    speakers = torch.tensor(speakers)
    genders = torch.tensor(genders)
    emotions = torch.tensor(emotions)
    intensities = torch.tensor(intensities)
    repetitions = torch.tensor(repetitions)
    acoustic_lengths = torch.tensor(acoustic_lengths)

    if not add_avging:
        acoustic_holder = nn.utils.rnn.pad_sequence(
            acoustic_holder, batch_first=True, padding_value=0
        )
    else:
        acoustic_holder = torch.stack(acoustic_holder)
        print(acoustic_holder.shape)

    # get means, stdev
    acoustic_means, acoustic_stdev = get_acoustic_means(acoustic_holder)

    if as_dict:
        for i in range(len(acoustic_holder)):
            acoustic_data = transform_acoustic_item(
                acoustic_holder[i], acoustic_means, acoustic_stdev
            )
            data.append(
                {
                    "x_acoustic": acoustic_data.clone().detach(),
                    "x_utt": utterances[i].clone().detach(),
                    "x_speaker": speakers[i].clone().detach(),
                    "x_gender": genders[i].clone().detach(),
                    "ys": [
                        intensities[i].clone().detach(),
                        emotions[i].clone().detach(),
                    ],
                    "repetition": repetitions[i].clone().detach(),
                    "utt_length": utt_length,
                    "acoustic_length": acoustic_lengths[i].clone().detach(),
                    "audio_id": audio_ids[i],
                }
            )
    else:
        for i in range(len(acoustic_holder)):
            acoustic_data = transform_acoustic_item(
                acoustic_holder[i], acoustic_means, acoustic_stdev
            )
            data.append(
                (
                    acoustic_data.clone().detach(),
                    utterances[i].clone().detach(),
                    speakers[i].clone().detach(),
                    genders[i].clone().detach(),
                    intensities[i].clone().detach(),
                    emotions[i].clone().detach(),
                    repetitions[i].clone().detach(),
                    6,
                    acoustic_lengths[i].clone().detach(),
                )
            )

    return data


def make_ravdess_data_tensors_with_custom_acoustic_features(
    acoustic_path,
    custom_feats_file,
    glove=None,
    use_cols=None,
    as_dict=False,
    selected_ids=None,
    bert_type="distilbert",
):
    """
    makes data tensors for use in RAVDESS objects
    acoustic_path: path to acoustic RAVDESS base dir
    custom_feats_file: the name of a file with custom features for entire dataset
    use_cols: if set, should be a list [] of column names to include
    selected_ids: if not none, should contain the order of all ids
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
    audio_ids = []

    # holder for all data
    data = []

    if glove is not None:
        utt_1 = glove.index(["kids", "are", "talking", "by", "the", "door"])
        utt_2 = glove.index(["dogs", "are", "sitting", "by", "the", "door"])
        utt_length = 6
    else:
        # instantiate embeddings maker
        if bert_type.lower() == "bert":
            emb_maker = BertEmb()
        else:
            emb_maker = DistilBertEmb()
        utt_1, id_1 = emb_maker.tokenize("kids are talking by the door")
        utt_1 = emb_maker.get_embeddings(utt_1, torch.tensor(id_1).unsqueeze(0), 8)
        utt_2, id_2 = emb_maker.tokenize("dogs are sitting by the door")
        utt_2 = emb_maker.get_embeddings(utt_2, torch.tensor(id_2).unsqueeze(0), 8)
        utt_length = max(len(utt_1), len(utt_2))

    # will have to do two for loops
    # one to get the longest acoustic df
    # the other to organize data tensors

    # open acoustic features file
    path_to_acoustic_file = f"{acoustic_path}/{custom_feats_file}"

    # read in features file as pd DataFrame
    if use_cols is not None:
        feats = pd.read_csv(path_to_acoustic_file, usecols=use_cols, sep=",")
    else:
        feats = pd.read_csv(path_to_acoustic_file, sep=",")

    feats.set_index("file_name", inplace=True)

    # extract necessary information from each row
    for index, row in feats.iterrows():
        labels_list = index.split("-")

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

        # order of items: acoustic, utt, spkr, gender, emotion
        #   intensity, repetition #, utt_length, acoustic_length
        # custom feature values are already averaged, so no need to
        #   add torch.mean here
        acoustic_holder.append(torch.tensor(row.tolist()))
        utterances.append(utt)
        speakers.append(speaker)
        genders.append(gender)
        emotions.append(emotion)
        intensities.append(intensity)
        repetitions.append(repetition)
        acoustic_lengths.append(1)  # because averaged, all are 1
        audio_ids.append(index)

    # convert data to torch tensors
    if glove is not None:
        utterances = torch.tensor(utterances)
    else:
        utterances = torch.stack(utterances)
        utterances = torch.squeeze(utterances, dim=0)
    speakers = torch.tensor(speakers)
    genders = torch.tensor(genders)
    emotions = torch.tensor(emotions)
    intensities = torch.tensor(intensities)
    repetitions = torch.tensor(repetitions)
    acoustic_lengths = torch.tensor(acoustic_lengths)

    # acoustic feats are already overall for file
    #   so don't need padding for an rnn
    acoustic_holder = torch.stack(acoustic_holder)
    print(acoustic_holder.shape)

    # get means, stdev
    acoustic_means, acoustic_stdev = get_acoustic_means(acoustic_holder)

    if as_dict:
        for i in range(len(acoustic_holder)):
            acoustic_data = transform_acoustic_item(
                acoustic_holder[i], acoustic_means, acoustic_stdev
            )
            data.append(
                {
                    "x_acoustic": acoustic_data.clone().detach(),
                    "x_utt": utterances[i].clone().detach(),
                    "x_speaker": speakers[i].clone().detach(),
                    "x_gender": genders[i].clone().detach(),
                    "ys": [
                        intensities[i].clone().detach(),
                        emotions[i].clone().detach(),
                    ],
                    "repetition": repetitions[i].clone().detach(),
                    "utt_length": utt_length,
                    "acoustic_length": acoustic_lengths[i].clone().detach(),
                    "audio_id": audio_ids[i],
                }
            )
    else:
        for i in range(len(acoustic_holder)):
            acoustic_data = transform_acoustic_item(
                acoustic_holder[i], acoustic_means, acoustic_stdev
            )
            data.append(
                (
                    acoustic_data.clone().detach(),
                    utterances[i].clone().detach(),
                    speakers[i].clone().detach(),
                    genders[i].clone().detach(),
                    intensities[i].clone().detach(),
                    emotions[i].clone().detach(),
                    repetitions[i].clone().detach(),
                    6,
                    acoustic_lengths[i].clone().detach(),
                )
            )

    # if using an ordered list of items
    if selected_ids:
        # find order of all ids in data
        data_ordered_ids = [item["audio_id"] for item in data]
        idx = list(range(len(data_ordered_ids)))
        # create dict of id: idx
        data2idx = dict(zip(data_ordered_ids, idx))
        # reorder data to match order of selected_ids
        new_data = []
        for item in selected_ids:
            new_data.append(data[data2idx[item]])
        data = new_data

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
