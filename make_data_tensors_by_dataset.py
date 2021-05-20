from utils.data_prep_helpers import clean_up_word

import torch
from torch import nn

def make_data_tensors_meld(text_data, longest_utt, glove, tokenizer):
    """
    Make the data tensors for meld
    :param text_data: a pandas df containing text and gold
    :param longest_utt: length of longest utt
    :param glove: an instance of class Glove
    :param tokenizer: a tokenizer
    :return: a dict containing tensors for utts, speakers, ys,
        and utterance lengths
    """
    # create holders for the data
    all_data = {"all_utts": [], "all_speakers": [], "all_emotions": [],
                "all_sentiments": [], "utt_lengths": [],
                "all_audio_ids": [], }

    for idx, row in text_data.iterrows():
        # get audio id
        all_data["all_audio_ids"].append(row["DiaID_UttID"])

        # create utterance-level holders
        utts = [0] * longest_utt

        # get values from row
        utt = tokenizer(clean_up_word(str(row['utterance'])))
        all_data["utt_lengths"].append(len(utt))

        spk_id = row["Speaker"]
        emo = row["Emotion"]
        sent = row["Sentiment"]

        # convert words to indices for glove
        utt_indexed = glove.index(utt)
        for i, item in enumerate(utt_indexed):
            utts[i] = item

        all_data["all_utts"].append(torch.tensor(utts))
        all_data["all_speakers"].append([spk_id])
        all_data["all_emotions"].append(emo)
        all_data["all_sentiments"].append(sent)

    # create pytorch tensors for each
    all_data["all_speakers"] = torch.tensor(all_data["all_speakers"])
    all_data["all_emotions"] = torch.tensor(all_data["all_emotions"])
    all_data["all_sentiments"] = torch.tensor(all_data["all_sentiments"])

    # pad and transpose utterance sequences
    all_data["all_utts"] = nn.utils.rnn.pad_sequence(all_data["all_utts"])
    all_data["all_utts"] = all_data["all_utts"].transpose(0, 1)

    # return data
    return all_data


def make_data_tensors_mustard(text_data, longest_utt, glove, tokenizer):
    """
    Make the data tensors for meld
    :param text_data: a pandas df containing text and gold
    :param longest_utt: length of longest utt
    :param glove: an instance of class Glove
    :param tokenizer: a tokenizer
    :return: a dict containing tensors for utts, speakers, ys,
        and utterance lengths
    """
    # create holders for the data
    all_data = {"all_utts": [], "all_speakers": [],
                "all_sarcasm": [], "utt_lengths": [],
                "all_audio_ids": [], }

    for idx, row in text_data.iterrows():
        # get audio id
        all_data["all_audio_ids"].append(row["clip_id"])

        # create utterance-level holders
        utts = [0] * longest_utt

        # get values from row
        utt = tokenizer(clean_up_word(str(row['utterance'])))
        all_data["utt_lengths"].append(len(utt))

        spk_id = row["speaker"]
        gend_id = row["gender"]
        sarc = row["sarcasm"]

        # convert words to indices for glove
        utt_indexed = glove.index(utt)
        for i, item in enumerate(utt_indexed):
            utts[i] = item

        all_data["all_utts"].append(torch.tensor(utts))
        all_data["all_speakers"].append(spk_id)
        all_data["all_genders"].append(gend_id)
        all_data["all_sarcasm"].append(sarc)

    # create pytorch tensors for each
    all_data["all_speakers"] = torch.tensor(all_data["all_speakers"])
    all_data["all_emotions"] = torch.tensor(all_data["all_emotions"])
    all_data["all_sarcasm"] = torch.tensor(all_data["all_sarcasm"])

    # pad and transpose utterance sequences
    all_data["all_utts"] = nn.utils.rnn.pad_sequence(all_data["all_utts"])
    all_data["all_utts"] = all_data["all_utts"].transpose(0, 1)

    # return data
    return all_data


def make_data_tensors_chalearn(text_data, longest_utt, glove, tokenizer):
    """
    Make the data tensors for meld
    :param text_data: a pandas df containing text and gold
    :param longest_utt: length of longest utt
    :param glove: an instance of class Glove
    :param tokenizer: a tokenizer
    :return: a dict containing tensors for utts, speakers, ys,
        and utterance lengths
    """
    # create holders for the data
    all_data = {"all_utts": [], "all_genders": [], "all_ethnicities": [],
                "all_extraversion": [], "all_neuroticism": [],
                "all_agreeableness": [], "all_openness": [],
                "all_conscientiousness": [], "all_interview": [],
                "all_audio_ids": [], "utt_lengths": []}

    for idx, row in text_data.iterrows():
        # get audio id
        all_data["all_audio_ids"].append(row["file"])

        # create utterance-level holders
        utts = [0] * longest_utt

        # get values from row
        utt = tokenizer(clean_up_word(str(row['utterance'])))
        all_data["utt_lengths"].append(len(utt))

        spk_id = row["speaker"]
        gend_id = row["gender"]
        sarc = row["sarcasm"]

        # convert words to indices for glove
        utt_indexed = glove.index(utt)
        for i, item in enumerate(utt_indexed):
            utts[i] = item

        all_data["all_utts"].append(torch.tensor(utts))
        all_data["all_speakers"].append(spk_id)
        all_data["all_genders"].append(gend_id)
        all_data["all_sarcasm"].append(sarc)

    # create pytorch tensors for each
    all_data["all_speakers"] = torch.tensor(all_data["all_speakers"])
    all_data["all_emotions"] = torch.tensor(all_data["all_emotions"])
    all_data["all_sarcasm"] = torch.tensor(all_data["all_sarcasm"])

    # pad and transpose utterance sequences
    all_data["all_utts"] = nn.utils.rnn.pad_sequence(all_data["all_utts"])
    all_data["all_utts"] = all_data["all_utts"].transpose(0, 1)

    # return data
    return all_data