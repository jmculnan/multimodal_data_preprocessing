from utils.data_prep_helpers import clean_up_word
from bert.prepare_bert_embeddings import DistilBertEmb, BertEmb

import torch
from torch import nn
from tqdm import tqdm


def make_data_tensors_meld(
    text_data,
    used_utts_list,
    longest_utt,
    tokenizer,
    glove=None,
    bert_type="distilbert",
):
    """
    Make the data tensors for meld
    :param text_data: a pandas df containing text and gold
    :param used_utts_list: a list of all utterances with acoustic data
    :param longest_utt: length of longest utt
    :param tokenizer: a tokenizer
    :param glove: an instance of class Glove
    :return: a dict containing tensors for utts, speakers, ys,
        and utterance lengths
    """
    # create holders for the data
    all_data = {
        "all_utts": [],
        "all_speakers": [],
        "all_emotions": [],
        "all_sentiments": [],
        "utt_lengths": [],
        "all_audio_ids": [],
    }

    if bert_type.lower() == "bert":
        emb_maker = BertEmb()
    elif bert_type.lower() == "roberta":
        emb_maker = BertEmb(use_roberta=True)
    else:
        emb_maker = DistilBertEmb()

    for idx, row in tqdm(text_data.iterrows(), total=len(text_data), desc="Organizing data for MELD"):
        # check if this item has acoustic data
        dia_num, utt_num = row["diaid_uttid"].split("_")[:2]
        if (dia_num, utt_num) in used_utts_list:
            # get audio id
            all_data["all_audio_ids"].append(row["diaid_uttid"])

            # get values from row
            if glove is not None:
                # create utterance-level holders
                utts = [0] * longest_utt

                # get glove indices if using glove
                utt = tokenizer(clean_up_word(str(row["utterance"])))
                all_data["utt_lengths"].append(len(utt))

                # convert words to indices for glove
                utt_indexed = glove.index(utt)
                for i, item in enumerate(utt_indexed):
                    utts[i] = item

                all_data["all_utts"].append(torch.tensor(utts))

            else:
                # else use the bert/distilbert tokenizer instead
                utt, ids = emb_maker.tokenize(clean_up_word(str(row["utterance"])))
                # convert ids to tensor
                ids = torch.tensor(ids)
                # bert requires an extra dimension to match utt
                if bert_type.lower() == "bert":
                    ids = ids.unsqueeze(0)
                utt_embs = emb_maker.get_embeddings(utt, ids, longest_utt)

                all_data["utt_lengths"].append(len(ids))

                all_data["all_utts"].append(utt_embs)

            spk_id = row["speaker"]
            emo = row["emotion"]
            sent = row["sentiment"]

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


def make_data_tensors_mustard(
    text_data,
    used_utts_list,
    longest_utt,
    tokenizer,
    glove=None,
    bert_type="distilbert",
):
    """
    Make the data tensors for meld
    :param text_data: a pandas df containing text and gold
    :param used_utts_list: a list of all utts with acoustic data
    :param longest_utt: length of longest utt
    :param tokenizer: a tokenizer
    :param glove: an instance of class Glove
    :return: a dict containing tensors for utts, speakers, ys,
        and utterance lengths
    """
    # create holders for the data
    all_data = {
        "all_utts": [],
        "all_speakers": [],
        "all_genders": [],
        "all_sarcasm": [],
        "utt_lengths": [],
        "all_audio_ids": [],
    }

    if bert_type.lower() == "bert":
        emb_maker = BertEmb()
    elif bert_type.lower() == "roberta":
        emb_maker = BertEmb(use_roberta=True)
    else:
        emb_maker = DistilBertEmb()

    for idx, row in tqdm(text_data.iterrows(), total=len(text_data), desc="Organizing data for MUStARD"):
        # check if this is in the list
        if row["clip_id"] in used_utts_list:

            # get audio id
            all_data["all_audio_ids"].append(row["clip_id"])

            # get values from row
            if glove is not None:
                # create utterance-level holders
                utts = [0] * longest_utt

                utt = tokenizer(clean_up_word(str(row["utterance"])))
                all_data["utt_lengths"].append(len(utt))

                # convert words to indices for glove
                utt_indexed = glove.index(utt)
                for i, item in enumerate(utt_indexed):
                    utts[i] = item

                all_data["all_utts"].append(torch.tensor(utts))

            else:
                # else use the bert/distilbert tokenizer instead
                utt, ids = emb_maker.tokenize(clean_up_word(str(row["utterance"])))
                # convert ids to tensor
                ids = torch.tensor(ids)
                # bert requires an extra dimension to match utt
                if bert_type.lower() == "bert":
                    ids = ids.unsqueeze(0)
                utt_embs = emb_maker.get_embeddings(utt, ids, longest_utt)

                all_data["utt_lengths"].append(len(ids))

                all_data["all_utts"].append(utt_embs)

            spk_id = row["speaker"]
            gend_id = row["gender"]
            sarc = row["sarcasm"]

            all_data["all_speakers"].append(spk_id)
            all_data["all_genders"].append(gend_id)
            all_data["all_sarcasm"].append(sarc)

    # create pytorch tensors for each
    all_data["all_genders"] = torch.tensor(all_data["all_genders"])
    all_data["all_sarcasm"] = torch.tensor(all_data["all_sarcasm"])

    # pad and transpose utterance sequences
    all_data["all_utts"] = nn.utils.rnn.pad_sequence(all_data["all_utts"])
    all_data["all_utts"] = all_data["all_utts"].transpose(0, 1)

    # return data
    return all_data


def make_data_tensors_chalearn(
    text_data,
    used_utts_list,
    longest_utt,
    tokenizer,
    glove=None,
    bert_type="distilbert",
):
    """
    Make the data tensors for meld
    :param text_data: a pandas df containing text and gold
    :param used_utts_list: the list of all utts with acoustic data
    :param longest_utt: length of longest utt
    :param tokenizer: a tokenizer
    :param glove: an instance of class Glove
    :return: a dict containing tensors for utts, speakers, ys,
        and utterance lengths
    """
    # create holders for the data
    all_data = {
        "all_utts": [],
        "all_genders": [],
        "all_ethnicities": [],
        "all_extraversion": [],
        "all_neuroticism": [],
        "all_agreeableness": [],
        "all_openness": [],
        "all_conscientiousness": [],
        "all_interview": [],
        "all_audio_ids": [],
        "utt_lengths": [],
    }

    if bert_type.lower() == "bert":
        emb_maker = BertEmb()
    elif bert_type.lower() == "roberta":
        emb_maker = BertEmb(use_roberta=True)
    else:
        emb_maker = DistilBertEmb()

    for idx, row in tqdm(text_data.iterrows(), total=len(text_data), desc="Organizing data for FirstImpr"):
        # check if this item has acoustic data
        audio_name = row["file"].split(".mp4")[0]
        if audio_name in used_utts_list:

            # get audio id
            all_data["all_audio_ids"].append(row["file"])

            # get values from row
            if glove is not None:
                # create utterance-level holders
                utts = [0] * longest_utt

                utt = tokenizer(clean_up_word(str(row["utterance"])))
                all_data["utt_lengths"].append(len(utt))

                # convert words to indices for glove
                utt_indexed = glove.index(utt)
                for i, item in enumerate(utt_indexed):
                    utts[i] = item

                all_data["all_utts"].append(torch.tensor(utts))
            else:
                # else use the bert/distilbert tokenizer instead
                utt, ids = emb_maker.tokenize(clean_up_word(str(row["utterance"])))
                # convert ids to tensor
                ids = torch.tensor(ids)
                # bert requires an extra dimension to match utt
                if bert_type.lower() == "bert":
                    ids = ids.unsqueeze(0)
                utt_embs = emb_maker.get_embeddings(utt, ids, longest_utt)

                all_data["utt_lengths"].append(len(ids))

                all_data["all_utts"].append(utt_embs)

            gend_id = row["gender"]
            eth_id = row["ethnicity"]
            extr_id = row["extraversion"]
            neur_id = row["neuroticism"]
            agree_id = row["agreeableness"]
            openn_id = row["openness"]
            consc_id = row["conscientiousness"]
            int_id = row["invite_to_interview"]

            all_data["all_genders"].append(gend_id)
            all_data["all_ethnicities"].append(eth_id)
            all_data["all_extraversion"].append(extr_id)
            all_data["all_neuroticism"].append(neur_id)
            all_data["all_agreeableness"].append(agree_id)
            all_data["all_openness"].append(openn_id)
            all_data["all_conscientiousness"].append(consc_id)
            all_data["all_interview"].append(int_id)

    # create pytorch tensors for each
    all_data["all_genders"] = torch.tensor(all_data["all_genders"])
    all_data["all_ethnicities"] = torch.tensor(all_data["all_ethnicities"])
    all_data["all_extraversion"] = torch.tensor(all_data["all_extraversion"])
    all_data["all_neuroticism"] = torch.tensor(all_data["all_neuroticism"])
    all_data["all_agreeableness"] = torch.tensor(all_data["all_agreeableness"])
    all_data["all_openness"] = torch.tensor(all_data["all_openness"])
    all_data["all_conscientiousness"] = torch.tensor(all_data["all_conscientiousness"])
    all_data["all_interview"] = torch.tensor(all_data["all_interview"])

    # pad and transpose utterance sequences
    all_data["all_utts"] = nn.utils.rnn.pad_sequence(all_data["all_utts"])
    all_data["all_utts"] = all_data["all_utts"].transpose(0, 1)

    # return data
    return all_data


def make_data_tensors_cdc(
    text_data,
    used_utts_list,
    longest_utt,
    tokenizer,
    glove=None,
    bert_type="distilbert",
):
    """
    Make the data tensors for meld
    :param text_data: a pandas df containing text and gold
    :param used_utts_list: the list of all utts with acoustic data
    :param longest_utt: length of longest utt
    :param tokenizer: a tokenizer
    :param glove: an instance of class Glove
    :return: a dict containing tensors for utts, speakers, ys,
        and utterance lengths
    """
    # create holders for the data
    all_data = {
        "all_utts": [],
        "all_truth_values": [],
        "all_speakers": [],
        "all_audio_ids": [],
        "utt_lengths": [],
    }

    if bert_type.lower() == "bert":
        emb_maker = BertEmb()
    elif bert_type.lower() == "roberta":
        emb_maker = BertEmb(use_roberta=True)
    else:
        emb_maker = DistilBertEmb()

    for idx, row in tqdm(text_data.iterrows(), total=len(text_data), desc="Organizing data for CDC"):
        # check if this item has acoustic data
        audio_name = str(row["utt_num"])

        if audio_name in used_utts_list:

            # get audio id
            all_data["all_audio_ids"].append(row["utt_num"])

            # get values from row
            if glove is not None:
                # create utterance-level holders
                utts = [0] * longest_utt

                utt = tokenizer(clean_up_word(str(row["utterance"])))
                all_data["utt_lengths"].append(len(utt))

                # convert words to indices for glove
                utt_indexed = glove.index(utt)
                for i, item in enumerate(utt_indexed):
                    utts[i] = item

                all_data["all_utts"].append(torch.tensor(utts))
            else:
                # else use the bert/distilbert tokenizer instead
                utt, ids = emb_maker.tokenize(clean_up_word(str(row["utterance"])))
                # convert ids to tensor
                ids = torch.tensor(ids)
                # bert requires an extra dimension to match utt
                if bert_type.lower() == "bert":
                    ids = ids.unsqueeze(0)
                utt_embs = emb_maker.get_embeddings(utt, ids, longest_utt)

                all_data["utt_lengths"].append(len(ids))

                all_data["all_utts"].append(utt_embs)

            spk_id = row["speaker"]
            truth_val = row["truth_value"]
            if truth_val == "TRUTH":
                truth_val = 1
            else:
                truth_val = 0

            all_data["all_speakers"].append(spk_id)
            all_data["all_truth_values"].append(truth_val)

    # create pytorch tensors for each
    all_data["all_truth_values"] = torch.tensor(all_data["all_truth_values"])

    # pad and transpose utterance sequences
    all_data["all_utts"] = nn.utils.rnn.pad_sequence(all_data["all_utts"])
    all_data["all_utts"] = all_data["all_utts"].transpose(0, 1)

    # return data
    return all_data


def make_data_tensors_mosi(
    text_data, used_utts_list, longest_utt, tokenizer, glove, bert_type="distilbert"
):
    """
    Make the data tensors for meld
    :param text_data: a pandas df containing text and gold
    :param used_utts_list: the list of all utts with acoustic data
    :param longest_utt: length of longest utt
    :param tokenizer: a tokenizer
    :param glove: an instance of class Glove
    :return: a dict containing tensors for utts, speakers, ys,
        and utterance lengths
    """
    # create holders for the data
    all_data = {
        "all_utts": [],
        "all_sentiments": [],
        "all_speakers": [],
        "all_audio_ids": [],
        "utt_lengths": [],
    }

    if bert_type.lower() == "bert":
        emb_maker = BertEmb()
    elif bert_type.lower() == "roberta":
        emb_maker = BertEmb(use_roberta=True)
    else:
        emb_maker = DistilBertEmb()

    for idx, row in tqdm(text_data.iterrows(), total=len(text_data), desc="Organizing data for MOSI"):
        if row["id"] in used_utts_list:

            # get audio id
            all_data["all_audio_ids"].append(row["id"])

            if glove is not None:
                # create utterance-level holders
                utts = [0] * longest_utt

                # get values from row
                utt = tokenizer(clean_up_word(str(row["utterance"])))
                all_data["utt_lengths"].append(len(utt))

                # convert words to indices for glove
                utt_indexed = glove.index(utt)
                for i, item in enumerate(utt_indexed):
                    utts[i] = item

                all_data["all_utts"].append(torch.tensor(utts))
            else:
                # else use the bert/distilbert tokenizer instead
                utt, ids = emb_maker.tokenize(clean_up_word(str(row["utterance"])))
                # convert ids to tensor
                ids = torch.tensor(ids)
                # bert requires an extra dimension to match utt
                if bert_type.lower() == "bert":
                    ids = ids.unsqueeze(0)
                utt_embs = emb_maker.get_embeddings(utt, ids, longest_utt)

                all_data["utt_lengths"].append(len(ids))

                all_data["all_utts"].append(utt_embs)

            spk_id = row["speaker"]
            sentiment = row["sentiment"]

            all_data["all_speakers"].append(spk_id)
            all_data["all_sentiments"].append(sentiment)

    # create pytorch tensors for each
    all_data["all_sentiments"] = torch.tensor(all_data["all_sentiments"])

    # pad and transpose utterance sequences
    all_data["all_utts"] = nn.utils.rnn.pad_sequence(all_data["all_utts"])
    all_data["all_utts"] = all_data["all_utts"].transpose(0, 1)

    # return data
    return all_data


def make_data_tensors_lives(
    text_data, used_utts_list, longest_utt, tokenizer, glove, bert_type="distilbert"
):
    """
    Make the data tensors for lives
    :param text_data: a pandas df containing text and gold
    :param used_utts_list: the list of all utts with acoustic data
    :param longest_utt: length of longest utt
    :param tokenizer: a tokenizer
    :param glove: an instance of class Glove
    :return: a dict containing tensors for utts, speakers, ys,
        and utterance lengths
    todo: background info + gold labels will NOT be included in these tensors
        it will be incorporated by being queried from separate location
        when these items are called during training and testing of models
    """
    # create holders for the data
    all_data = {
        "all_utts": [],
        "all_speakers": [],
        "all_audio_ids": [],
        "utt_lengths": [],
        "all_recording_ids": [],
        "all_call_ids": [],
        "all_utt_nums": [],
    }

    if bert_type.lower() == "bert":
        emb_maker = BertEmb()
    elif bert_type.lower() == "roberta":
        emb_maker = BertEmb(use_roberta=True)
    else:
        emb_maker = DistilBertEmb()

    for idx, row in tqdm(text_data.iterrows(), total=len(text_data), desc="Organizing data for LIvES"):
        # if f"{row['recording_id']}_{row['utt_num']}" in used_utts_list:
        if f"{row['recording_id']}_utt{row['utt_num']}_speaker{row['speaker']}" in used_utts_list:
        # if row["id"] in used_utts_list:

            # get audio id
            all_data["all_audio_ids"].append(f"{row['recording_id']}_utt{row['utt_num']}_speaker{row['speaker']}")

            if glove is not None:
                # create utterance-level holders
                utts = [0] * longest_utt

                # get values from row
                utt = tokenizer(clean_up_word(str(row["utterance"])))
                all_data["utt_lengths"].append(len(utt))

                # convert words to indices for glove
                utt_indexed = glove.index(utt)
                for i, item in enumerate(utt_indexed):
                    utts[i] = item

                all_data["all_utts"].append(torch.tensor(utts))
            else:
                # else use the bert/distilbert tokenizer instead
                utt, ids = emb_maker.tokenize(clean_up_word(str(row["utterance"])))

                # convert ids to tensor
                ids = torch.tensor(ids)
                all_data["utt_lengths"].append(len(ids))

                # bert requires an extra dimension to match utt
                if bert_type.lower() == "bert":
                    ids = ids.unsqueeze(0)
                utt_embs = emb_maker.get_embeddings(utt, ids, longest_utt)

                all_data["all_utts"].append(utt_embs)

            spk_id = f"{row['speaker']}-{row['sid']}"

            recording_id = row['recording_id']
            utt_num = row['utt_num']

            all_data["all_speakers"].append(spk_id)
            all_data["all_recording_ids"].append(recording_id)
            all_data["all_utt_nums"].append(utt_num)

    # pad and transpose utterance sequences
    all_data["all_utts"] = nn.utils.rnn.pad_sequence(all_data["all_utts"])
    all_data["all_utts"] = all_data["all_utts"].transpose(0, 1)

    # return data
    return all_data


def make_data_tensors_asist(
    text_data, used_utts_list, longest_utt, tokenizer, glove, bert_type="distilbert", use_text=True
):
    """
    Make the data tensors for asist
    :param text_data: a pandas df containing text and gold
    :param used_utts_list: the list of all utts with acoustic data
    :param longest_utt: length of longest utt
    :param tokenizer: a tokenizer
    :param glove: an instance of class Glove
    :param bert_type: the type of bert embeddings to get, if using
    :param use_text: whether to save raw text instead of embeddings
    :return: a dict containing tensors for utts, speakers, ys,
        and utterance lengths
    """
    # create holders for the data
    all_data = {
        "all_utts": [],
        "all_sentiments": [],
        "all_emotions": [],
        "all_traits": [],
        "all_speakers": [],
        "all_audio_ids": [],
        "utt_lengths": [],
    }

    # set class numbers for sent, emo
    sent2idx = {'positive': 2, 'neutral': 1, 'negative': 0}
    emo2idx = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4,
               'sadness': 5, 'surprise': 6}
    trait2idx = {'extroversion': 0, 'neuroticism': 1, 'agreeableness': 2,
                 'openness': 3, 'conscientiousness': 4}

    if bert_type.lower() == "bert":
        emb_maker = BertEmb()
    elif bert_type.lower() == "roberta":
        emb_maker = BertEmb(use_roberta=True)
    else:
        emb_maker = DistilBertEmb()

    for idx, row in tqdm(text_data.iterrows(), total=len(text_data), desc="Organizing data for MultiCAT"):
        if row["message_id"] in used_utts_list:

            # get audio id
            all_data["all_audio_ids"].append(row["message_id"])

            if glove is not None:
                # create utterance-level holders
                utts = [0] * longest_utt

                # get values from row
                utt = tokenizer(clean_up_word(str(row["utt"])))
                all_data["utt_lengths"].append(len(utt))

                # convert words to indices for glove
                utt_indexed = glove.index(utt)
                for i, item in enumerate(utt_indexed):
                    utts[i] = item

                all_data["all_utts"].append(torch.tensor(utts))
            elif bert_type.lower() == "text":
                # get values from row
                utt = tokenizer(clean_up_word(str(row['utt'])))
                utt.insert(0, "[CLS]")
                utt.append("[SEP]")

                all_data["utt_lengths"].append(len(utt))
                all_data["all_utts"].append(utt)
            else:
                # else use the bert/distilbert tokenizer instead
                utt, ids = emb_maker.tokenize(clean_up_word(str(row["utt"])))
                # convert ids to tensor
                ids = torch.tensor(ids)
                all_data["utt_lengths"].append(len(ids))
                # bert requires an extra dimension to match utt
                if bert_type.lower() == "bert":
                    ids = ids.unsqueeze(0)
                utt_embs = emb_maker.get_embeddings(utt, ids, longest_utt)

                all_data["all_utts"].append(utt_embs)

            spk_id = row["participant"]
            sentiment = sent2idx[row["sentiment"]]
            emotion = emo2idx[row["emotion"]]

            # fixme
            trait = 0
            # trait = trait2idx[row["max_trait"]]

            all_data["all_speakers"].append(spk_id)
            all_data["all_sentiments"].append(sentiment)
            all_data["all_emotions"].append(emotion)
            all_data["all_traits"].append(trait)

    # create pytorch tensors for each
    all_data["all_sentiments"] = torch.tensor(all_data["all_sentiments"])
    all_data["all_emotions"] = torch.tensor(all_data["all_emotions"])
    all_data["all_traits"] = torch.tensor(all_data["all_traits"])

    # pad and transpose utterance sequences
    if not use_text:
        all_data["all_utts"] = nn.utils.rnn.pad_sequence(all_data["all_utts"])
        all_data["all_utts"] = all_data["all_utts"].transpose(0, 1)

    # return data
    return all_data
