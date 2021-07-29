# create a constructed dataset by combining individual datasets
# add task label to each

from scripts.save_partitioned_data import prep_data
from utils.data_prep_helpers import look_up_num_classes


def create_constructed_data(
        data_list,
        data_paths,
        feature_set,
        transcription_type,
        glove_path,
        feats_to_use=None,
        combine_partitions=True,
):
    """
    Get a constructed dataset containing all data
    :param data_list: A list of datasets to include
        if pred type is needed, that item should be
        a (dataset, predtype) double
    :param data_paths: list of paths to the datasets
    :return:
    """
    # set a counter for the number of classes
    classes = 0

    # set holder for combined data
    combined_data = []

    # get each dataset
    for i, dset in enumerate(data_list):
        # get the data folds
        if type(dset) == str:
            train, dev, test, clsswt = prep_data(dset, data_paths[i], feature_set, transcription_type, glove_path,
                                                 feats_to_use, data_as_dict=True)
            dset_name = dset

        elif type(dset) == tuple:
            train, dev, test, clsswt = prep_data(dset[0], data_paths[i], feature_set, transcription_type, glove_path,
                                                 feats_to_use, dset[1], data_as_dict=True)

            dset_name = f"{dset[0]}_{dset[1]}"

        # combine partitions if need be
        if combine_partitions:
            all_data = train + dev + test

        # get number of classes in this dataset
        num_classes = look_up_num_classes(dset_name)

        # alter the y values (class numbers)
        for item in all_data:
            # set the dataset number
            item["dataset_num"] = i
            # set the class number out of the pool of all classes
            item["ys"][0] = item["ys"][0] + classes
            # if num_classes is a list, though, there are 2 y values
            if type(num_classes) == list:
                # add both the number of classes in item 1 + classes counter
                item["ys"][1] = item["ys"][1] + classes + num_classes[0]

        # add updated data to all data holder
        combined_data.extend(all_data)

        # add number of classes seen to class counter
        if type(num_classes) == list:
            num_classes = sum(num_classes)

        classes += num_classes

    print(classes)
    print(combined_data[::10000])
    # return altered dataset
    return combined_data


def select_by_class(dataset, class_nums, gold_idx=4, data_is_dict=False, ys_idx=0):
    """
    Select a portion of a dataset consisting of a specific class number
    :param dataset: The dataset from which to extract classes
    :param class_nums: A list of class numbers to extract
    :param gold_idx: The index containing gold y value
    :param data_is_dict: Whether each datapoint is a dict
    :param ys_idx: the index of relevant task in the dataset ys list
    only relevant when data is a dict
    :return: a dataset composed only of desired classes
    """
    # create a dataset holder
    data_subset = []

    # for item in dataset
    for item in dataset:

        # if item[class_num_idx] in class_nums
        if data_is_dict:
            # ys is a key, val is a list of lists
            if item["ys"][ys_idx][gold_idx] in class_nums:
                data_subset.append(item)
        else:
            if item[gold_idx] in class_nums:
                # add to dataset holder
                data_subset.append(item)

    # return
    return data_subset


if __name__ == "__main__":
    base_path = "../../datasets/multimodal_datasets"
    cdc_path = f"{base_path}/Columbia_deception_corpus"
    mosi_path = f"{base_path}/CMU_MOSI"
    firstimpr_path = f"{base_path}/Chalearn"
    meld_path = f"{base_path}/MELD_formatted"
    mustard_path = f"{base_path}/MUStARD"
    ravdess_path = f"{base_path}/RAVDESS_Speech"

    datasets = ["cdc", ("mosi", "ternary"), ("firstimpr", "maxclass"),
                "meld", "mustard", "ravdess"]

    datasets_paths = [cdc_path, mosi_path, firstimpr_path, meld_path,
                      mustard_path, ravdess_path]

    glove_path = "../../datasets/glove/glove.subset.300d.txt"

    feature_set = "IS13"

    transcription_type = "gold"

    create_constructed_data(datasets, datasets_paths, feature_set,
                            transcription_type, glove_path)
