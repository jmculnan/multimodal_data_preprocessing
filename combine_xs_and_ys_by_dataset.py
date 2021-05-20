import torch

from utils.data_prep_helpers import transform_acoustic_item


def combine_xs_and_ys_meld(data_dict, acoustic_data, acoustic_lengths,
                               acoustic_means, acoustic_stdev):
    """
    Combine all x and y data into list of tuples for easier access with DataLoader
    """
    data = []

    for i, item in enumerate(acoustic_data):
        item_transformed = transform_acoustic_item(
            item, acoustic_means, acoustic_stdev
        )
        data.append(
            (
                item_transformed,
                data_dict["all_utts"][i],
                data_dict["all_speakers"][i],
                0,  # fixme: genders temporarily removed
                data_dict["all_emotions"][i],
                data_dict["all_sentiments"][i],
                data_dict["all_audio_ids"][i],
                data_dict["utt_lengths"][i],
                acoustic_lengths[i],
            )
        )

    return data


def combine_xs_and_ys_mustard(data_dict, acoustic_data, acoustic_lengths,
                               acoustic_means, acoustic_stdev):
    """
    Combine all x and y data into list of tuples for easier access with DataLoader
    """
    data = []

    for i, item in enumerate(acoustic_data):
        item_transformed = transform_acoustic_item(
            item, acoustic_means, acoustic_stdev
        )
        data.append(
            (
                item_transformed,
                data_dict["all_utts"][i],
                data_dict["all_speakers"][i],
                0,  # fixme: genders temporarily removed
                data_dict["all_sarcasm"][i],
                data_dict["all_audio_ids"][i],
                data_dict["utt_lengths"][i],
                acoustic_lengths[i],
            )
        )

    return data

def combine_xs_and_ys_chalearn(data_dict, acoustic_data, acoustic_lengths,
                               acoustic_means, acoustic_stdev, pred_type):
    """
    Combine all x and y data into list of tuples for easier access with DataLoader
    """
    data = []

    for i, item in enumerate(acoustic_data):
        item_transformed = transform_acoustic_item(
            item, acoustic_means, acoustic_stdev
        )
        if pred_type is not "max_class":
            data.append(
                (
                    item_transformed,
                    data_dict["all_utts"][i],
                    0,  # todo: eventually add speaker ?
                    data_dict["all_genders"][i],
                    data_dict["all_ethnicities"][i],
                    data_dict["all_extraversion"][i],
                    data_dict["all_neuroticism"][i],
                    data_dict["all_agreeableness"][i],
                    data_dict["all_openness"][i],
                    data_dict["all_conscientiousness"][i],
                    data_dict["all_interview"][i],
                    data_dict["all_audio_ids"][i],
                    data_dict["utt_lengths"][i],
                    acoustic_lengths[i],
                )
            )
        else:
            ys = [data_dict["all_extraversion"][i],
                  data_dict["all_neuroticism"][i],
                  data_dict["all_agreeableness"][i],
                  data_dict["all_openness"][i],
                  data_dict["all_conscientiousness"][i],]
            item_y = ys.index(max(ys))
            data.append((
                item_transformed,
                data_dict["all_utts"][i],
                0,  # todo: eventually add speaker ?
                data_dict["all_genders"][i],
                torch.tensor(item_y),
                data_dict["all_ethnicities"][i],
                data_dict["all_audio_ids"][i],
                data_dict["utt_lengths"][i],
                acoustic_lengths[i],
            )
            )

    return data
