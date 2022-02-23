import torch

from utils.data_prep_helpers import transform_acoustic_item


def combine_xs_and_ys_meld(
    data_dict,
    acoustic_data,
    acoustic_lengths,
    acoustic_means,
    acoustic_stdev,
    spec_data=None,
    spec_lengths=None,
    as_dict=False,
):
    """
    Combine all x and y data into list of tuples for easier access with DataLoader
    """
    data = []
    if as_dict:
        for i, item in enumerate(acoustic_data):
            item_transformed = transform_acoustic_item(
                item, acoustic_means, acoustic_stdev
            )
            data.append(
                {
                    "x_acoustic": item_transformed.clone().detach(),
                    "x_utt": data_dict["all_utts"][i].clone().detach(),
                    "x_spec": spec_data[i].clone().detach() if spec_data else 0,
                    "x_speaker": data_dict["all_speakers"][i].clone().detach(),
                    "x_gender": 0,
                    "ys": [
                        data_dict["all_emotions"][i].clone().detach(),
                        data_dict["all_sentiments"][i].clone().detach(),
                    ],
                    "audio_id": data_dict["all_audio_ids"][i],
                    "utt_length": data_dict["utt_lengths"][i],
                    "acoustic_length": acoustic_lengths[i],
                    "spec_length": spec_lengths[i] if spec_lengths else 0,
                }
            )
    else:
        for i, item in enumerate(acoustic_data):
            item_transformed = transform_acoustic_item(
                item, acoustic_means, acoustic_stdev
            )
            data.append(
                (
                    item_transformed.clone().detach(),
                    data_dict["all_utts"][i].clone().detach(),
                    data_dict["all_speakers"][i].clone().detach(),
                    0,  # fixme: genders temporarily removed
                    data_dict["all_emotions"][i].clone().detach(),
                    data_dict["all_sentiments"][i].clone().detach(),
                    spec_data[i].clone().detach() if spec_data else 0,
                    spec_lengths[i] if spec_lengths else 0,
                    data_dict["all_audio_ids"][i],
                    data_dict["utt_lengths"][i],
                    acoustic_lengths[i],
                )
            )

    return data


def combine_xs_and_ys_mustard(
    data_dict,
    acoustic_data,
    acoustic_lengths,
    acoustic_means,
    acoustic_stdev,
    speaker2idx,
    as_dict=False,
    spec_data=None,
    spec_lengths=None,
):
    """
    Combine all x and y data into list of tuples for easier access with DataLoader
    """
    data = []

    if as_dict:
        for i, item in enumerate(acoustic_data):
            item_transformed = transform_acoustic_item(
                item, acoustic_means, acoustic_stdev
            )
            data.append(
                {
                    "x_acoustic": item_transformed.clone().detach(),
                    "x_utt": data_dict["all_utts"][i].clone().detach(),
                    "x_spec": spec_data[i].clone().detach() if spec_data else 0,
                    "x_speaker": speaker2idx[data_dict["all_speakers"][i]],
                    "x_gender": 0,  # fixme: genders temporarily removed
                    "ys": [data_dict["all_sarcasm"][i].clone().detach()],
                    "audio_id": data_dict["all_audio_ids"][i],
                    "utt_length": data_dict["utt_lengths"][i],
                    "acoustic_length": acoustic_lengths[i],
                    "spec_length": spec_lengths[i] if spec_lengths else 0
                }
            )
    else:
        for i, item in enumerate(acoustic_data):
            item_transformed = transform_acoustic_item(
                item, acoustic_means, acoustic_stdev
            )
            data.append(
                (
                    item_transformed.clone().detach(),
                    data_dict["all_utts"][i].clone().detach(),
                    speaker2idx[data_dict["all_speakers"][i]],
                    0,  # fixme: genders temporarily removed
                    data_dict["all_sarcasm"][i].clone().detach(),
                    spec_data[i].clone().detach() if spec_data else 0,
                    spec_lengths[i] if spec_lengths else 0,
                    data_dict["all_audio_ids"][i],
                    data_dict["utt_lengths"][i],
                    acoustic_lengths[i],
                )
            )

    return data


def combine_xs_and_ys_firstimpr(
    data_dict,
    acoustic_data,
    acoustic_lengths,
    acoustic_means,
    acoustic_stdev,
    pred_type,
    as_dict=False,
    spec_data=None,
    spec_lengths=None,
):
    """
    Combine all x and y data into list of tuples for easier access with DataLoader
    """
    data = []

    if as_dict:
        for i, item in enumerate(acoustic_data):
            item_transformed = transform_acoustic_item(
                item, acoustic_means, acoustic_stdev
            )
            if pred_type is not "max_class":
                data.append(
                    {
                        "x_acoustic": item_transformed.clone().detach(),
                        "x_utt": data_dict["all_utts"][i].clone().detach(),
                        "x_spec": spec_data[i].clone().detach() if spec_data else 0,
                        "x_speaker": 0,  # todo: eventually add speaker ?
                        "x_gender": data_dict["all_genders"][i].clone().detach(),
                        "x_ethnicity": data_dict["all_ethnicities"][i].clone().detach(),
                        "ys": [
                            data_dict["all_extraversion"][i].clone().detach(),
                            data_dict["all_neuroticism"][i].clone().detach(),
                            data_dict["all_agreeableness"][i].clone().detach(),
                            data_dict["all_openness"][i].clone().detach(),
                            data_dict["all_conscientiousness"][i].clone().detach(),
                            data_dict["all_interview"][i].clone().detach(),
                        ],
                        "audio_id": data_dict["all_audio_ids"][i],
                        "utt_length": data_dict["utt_lengths"][i],
                        "acoustic_length": acoustic_lengths[i],
                        "spec_length": spec_lengths[i] if spec_lengths else 0,
                    }
                )
            else:
                ys = [
                    data_dict["all_extraversion"][i],
                    data_dict["all_neuroticism"][i],
                    data_dict["all_agreeableness"][i],
                    data_dict["all_openness"][i],
                    data_dict["all_conscientiousness"][i],
                ]
                item_y = ys.index(max(ys))
                data.append(
                    {
                        "x_acoustic": item_transformed.clone().detach(),
                        "x_utt": data_dict["all_utts"][i].clone().detach(),
                        "x_spec": spec_data[i].clone().detach() if spec_data else 0,
                        "x_speaker": 0,  # todo: eventually add speaker ?
                        "x_gender": data_dict["all_genders"][i].clone().detach(),
                        "ys": [torch.tensor(item_y)],
                        "x_ethnicity": data_dict["all_ethnicities"][i].clone().detach(),
                        "audio_id": data_dict["all_audio_ids"][i],
                        "utt_length": data_dict["utt_lengths"][i],
                        "acoustic_length": acoustic_lengths[i],
                        "spec_length": spec_lengths[i] if spec_lengths else 0,
                    }
                )

    else:
        for i, item in enumerate(acoustic_data):
            item_transformed = transform_acoustic_item(
                item, acoustic_means, acoustic_stdev
            )
            if pred_type is not "max_class":
                data.append(
                    (
                        item_transformed.clone().detach(),
                        data_dict["all_utts"][i].clone().detach(),
                        0,  # todo: eventually add speaker ?
                        data_dict["all_genders"][i].clone().detach(),
                        data_dict["all_ethnicities"][i].clone().detach(),
                        data_dict["all_extraversion"][i].clone().detach(),
                        data_dict["all_neuroticism"][i].clone().detach(),
                        data_dict["all_agreeableness"][i].clone().detach(),
                        data_dict["all_openness"][i].clone().detach(),
                        data_dict["all_conscientiousness"][i].clone().detach(),
                        data_dict["all_interview"][i].clone().detach(),
                        spec_data[i].clone().detach() if spec_data else 0,
                        spec_lengths[i] if spec_lengths else 0,
                        data_dict["all_audio_ids"][i],
                        data_dict["utt_lengths"][i],
                        acoustic_lengths[i],
                    )
                )
            else:
                ys = [
                    data_dict["all_extraversion"][i],
                    data_dict["all_neuroticism"][i],
                    data_dict["all_agreeableness"][i],
                    data_dict["all_openness"][i],
                    data_dict["all_conscientiousness"][i],
                ]
                item_y = ys.index(max(ys))
                data.append(
                    (
                        item_transformed.clone().detach(),
                        data_dict["all_utts"][i].clone().detach(),
                        0,  # todo: eventually add speaker ?
                        data_dict["all_genders"][i].clone().detach(),
                        torch.tensor(item_y),
                        data_dict["all_ethnicities"][i].clone().detach(),
                        spec_data[i].clone().detach() if spec_data else 0,
                        spec_lengths[i] if spec_lengths else 0,
                        data_dict["all_audio_ids"][i],
                        data_dict["utt_lengths"][i],
                        acoustic_lengths[i],
                    )
                )

    return data


def combine_xs_and_ys_cdc(
    data_dict,
    acoustic_data,
    acoustic_lengths,
    acoustic_means,
    acoustic_stdev,
    speaker2idx,
    as_dict=False,
    spec_data=None,
    spec_lengths=None,
):
    """
    Combine all x and y data into list of tuples for easier access with DataLoader
    """
    data = []

    if as_dict:
        for i, item in enumerate(acoustic_data):
            item_transformed = transform_acoustic_item(
                item, acoustic_means, acoustic_stdev
            )
            data.append(
                {
                    "x_acoustic": item_transformed.clone().detach(),
                    "x_utt": data_dict["all_utts"][i].clone().detach(),
                    "x_spec": spec_data[i].clone().detach() if spec_data else 0,
                    "x_speaker": speaker2idx[data_dict["all_speakers"][i]],
                    "x_gender": 0,  # todo: add gender later?
                    "ys": [data_dict["all_truth_values"][i].clone().detach()],
                    "audio_id": data_dict["all_audio_ids"][i],
                    "utt_length": data_dict["utt_lengths"][i],
                    "acoustic_length": acoustic_lengths[i],
                    "spec_length": spec_lengths[i] if spec_lengths else 0,
                }
            )

    else:
        for i, item in enumerate(acoustic_data):
            item_transformed = transform_acoustic_item(
                item, acoustic_means, acoustic_stdev
            )
            data.append(
                (
                    item_transformed.clone().detach(),
                    data_dict["all_utts"][i].clone().detach(),
                    speaker2idx[data_dict["all_speakers"][i]],
                    0,  # todo: add gender later?
                    data_dict["all_truth_values"][i].clone().detach(),
                    spec_data[i].clone().detach() if spec_data else 0,
                    spec_lengths[i] if spec_lengths else 0,
                    data_dict["all_audio_ids"][i],
                    data_dict["utt_lengths"][i],
                    acoustic_lengths[i],
                )
            )

    return data


def combine_xs_and_ys_mosi(
    data_dict,
    acoustic_data,
    acoustic_lengths,
    acoustic_means,
    acoustic_stdev,
    speaker2idx,
    pred_type,
    as_dict=False,
    spec_data=None,
    spec_lengths=None,
):
    """
    Combine all x and y data into list of tuples for easier access with DataLoader
    """
    data = []

    if as_dict:
        # to keep the numbers as they are
        if pred_type == "regression":
            for i, item in enumerate(acoustic_data):
                item_transformed = transform_acoustic_item(
                    item, acoustic_means, acoustic_stdev
                )
                data.append(
                    {
                        "x_acoustic": item_transformed.clone().detach(),
                        "x_utt": data_dict["all_utts"][i].clone().detach(),
                        "x_spec": spec_data[i].clone().detach() if spec_data else 0,
                        "x_speaker": speaker2idx[data_dict["all_speakers"][i]],
                        "x_gender": 0,  # todo: add gender later?
                        "ys": [data_dict["all_sentiments"][i].clone().detach()],
                        "audio_id": data_dict["all_audio_ids"][i],
                        "utt_length": data_dict["utt_lengths"][i],
                        "acoustic_length": acoustic_lengths[i],
                        "spec_length": spec_lengths[i] if spec_lengths else 0,
                    }
                )
        # to do a 7-class classification
        elif pred_type == "classification":
            sent2score = {-3: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6}
            for i, item in enumerate(acoustic_data):
                item_transformed = transform_acoustic_item(
                    item, acoustic_means, acoustic_stdev
                )
                data.append(
                    {
                        "x_acoustic": item_transformed.clone().detach(),
                        "x_utt": data_dict["all_utts"][i].clone().detach(),
                        "x_spec": spec_data[i].clone().detach() if spec_data else 0,
                        "x_speaker": speaker2idx[data_dict["all_speakers"][i]],
                        "x_gender": 0,  # todo: add gender later?
                        "ys": [
                            torch.tensor(
                                sent2score[round(data_dict["all_sentiments"][i].item())]
                            )
                        ],
                        "audio_id": data_dict["all_audio_ids"][i],
                        "utt_length": data_dict["utt_lengths"][i],
                        "acoustic_length": acoustic_lengths[i],
                        "spec_length": spec_lengths[i] if spec_lengths else 0,
                    }
                )
        # to do a 3-class classification
        elif pred_type == "ternary":
            for i, item in enumerate(acoustic_data):
                if data_dict["all_sentiments"][i] > 0:
                    sentiment_val = 2
                elif data_dict["all_sentiments"][i] == 0:
                    sentiment_val = 1
                else:
                    sentiment_val = 0
                item_transformed = transform_acoustic_item(
                    item, acoustic_means, acoustic_stdev
                )
                data.append(
                    {
                        "x_acoustic": item_transformed.clone().detach(),
                        "x_utt": data_dict["all_utts"][i].clone().detach(),
                        "x_spec": spec_data[i].clone().detach() if spec_data else 0,
                        "x_speaker": speaker2idx[data_dict["all_speakers"][i]],
                        "x_gender": 0,  # todo: add gender later?
                        "ys": [torch.tensor(sentiment_val)],
                        "audio_id": data_dict["all_audio_ids"][i],
                        "utt_length": data_dict["utt_lengths"][i],
                        "acoustic_length": acoustic_lengths[i],
                        "spec_length": spec_lengths[i] if spec_lengths else 0,
                    }
                )

    else:
        # to keep the numbers as they are
        if pred_type == "regression":
            for i, item in enumerate(acoustic_data):
                item_transformed = transform_acoustic_item(
                    item, acoustic_means, acoustic_stdev
                )
                data.append(
                    (
                        item_transformed.clone().detach(),
                        data_dict["all_utts"][i].clone().detach(),
                        speaker2idx[data_dict["all_speakers"][i]],
                        0,  # todo: add gender later?
                        data_dict["all_sentiments"][i].clone().detach(),
                        spec_data[i].clone().detach() if spec_data else 0,
                        spec_lengths[i] if spec_lengths else 0,
                        data_dict["all_audio_ids"][i],
                        data_dict["utt_lengths"][i],
                        acoustic_lengths[i],
                    )
                )
        # to do a 7-class classification
        elif pred_type == "classification":
            sent2score = {-3: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6}
            for i, item in enumerate(acoustic_data):
                item_transformed = transform_acoustic_item(
                    item, acoustic_means, acoustic_stdev
                )
                data.append(
                    (
                        item_transformed.clone().detach(),
                        data_dict["all_utts"][i].clone().detach(),
                        speaker2idx[data_dict["all_speakers"][i]],
                        0,  # todo: add gender later?
                        torch.tensor(
                            sent2score[round(data_dict["all_sentiments"][i].item())]
                        ),
                        spec_data[i].clone().detach() if spec_data else 0,
                        spec_lengths[i] if spec_lengths else 0,
                        data_dict["all_audio_ids"][i],
                        data_dict["utt_lengths"][i],
                        acoustic_lengths[i],
                    )
                )
        # to do a 3-class classification
        elif pred_type == "ternary":
            for i, item in enumerate(acoustic_data):
                if data_dict["all_sentiments"][i] > 0:
                    sentiment_val = 2
                elif data_dict["all_sentiments"][i] == 0:
                    sentiment_val = 1
                else:
                    sentiment_val = 0
                item_transformed = transform_acoustic_item(
                    item, acoustic_means, acoustic_stdev
                )
                data.append(
                    (
                        item_transformed.clone().detach(),
                        data_dict["all_utts"][i].clone().detach(),
                        speaker2idx[data_dict["all_speakers"][i]],
                        0,  # todo: add gender later?
                        torch.tensor(sentiment_val),
                        spec_data[i].clone().detach() if spec_data else 0,
                        spec_lengths[i] if spec_lengths else 0,
                        data_dict["all_audio_ids"][i],
                        data_dict["utt_lengths"][i],
                        acoustic_lengths[i],
                    )
                )
    return data


def combine_xs_and_ys_lives(
    data_dict,
    acoustic_data,
    acoustic_lengths,
    acoustic_means,
    acoustic_stdev,
    speaker2idx,
    as_dict=False,
    spec_data=None,
    spec_lengths=None,
):
    """
    Combine all x and y data into list of tuples for easier access with DataLoader
    """
    data = []

    if as_dict:
        for i, item in enumerate(acoustic_data):
            item_transformed = transform_acoustic_item(
                item, acoustic_means, acoustic_stdev
            )
            data.append(
                {
                    "x_acoustic": item_transformed.clone().detach(),
                    "x_utt": data_dict["all_utts"][i].clone().detach(),
                    "x_spec": spec_data[i].clone().detach() if spec_data else 0,
                    "x_speaker": speaker2idx[data_dict["all_speakers"][i]],
                    "x_gender": 0,
                    "audio_id": data_dict["all_audio_ids"][i],
                    "recording_id": data_dict["all_recording_ids"][i],
                    "utt_length": data_dict["utt_lengths"][i],
                    "acoustic_length": acoustic_lengths[i],
                    "spec_length": spec_lengths[i] if spec_lengths else 0,
                }
            )
    else:
        # to keep the numbers as they are
        exit("LIvES data can only be used in dict format")

    return data