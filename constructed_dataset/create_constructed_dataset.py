# create a constructed dataset by combining individual datasets
# add task label to each

import pickle
import random

# set random seed for replicability
random.seed(88)


class ConstructedDataset:
    def __init__(self, dataset, name=None):
        """
        :param dataset: either a path to a pickled data file or an unpickled dataset
        """
        if type(dataset) == str:
            self.data_name = dataset
            self.dataset = self.unpickle(dataset)
        else:
            self.dataset = dataset
            self.data_name = name
        self.data_is_dict = True if type(self.dataset[0]) == dict else False

        if self.data_is_dict:
            self.task_num = 0
        else:
            # get ys idx
            self.ys_idx = 4

    def unpickle(self, pickled):
        with open(pickled, "rb") as pfile:
            unpickled = pickle.load(pfile)

        return unpickled

    def change_task(self, new_num):
        # change task
        if self.data_is_dict:
            # use place in ys list if data as dict
            self.task_num = new_num
        else:
            # else use place in overall list
            self.ys_idx = new_num

    def add_class_nums(self, numslist):
        for num in numslist:
            self.class_nums.append(num)

    def create_constructed_data(self, class_nums):
        """
        Select a portion of a dataset consisting of a specific class number
        :param class_nums: a list of classes to select from a given dataset
        :return: a dataset composed only of desired classes
        """
        # create a dataset holder
        data_subset = []

        # for item in dataset
        for item in self.dataset:

            # if item[class_num_idx] in class_nums
            if self.data_is_dict:
                # ys is a key, val is a list of lists
                if item["ys"][self.task_num] in class_nums:
                    data_subset.append(item)
            else:
                if item[self.ys_idx] in class_nums:
                    # add to dataset holder
                    data_subset.append(item)

        # return
        return data_subset

    def save_constructed_task(self, constructed_task, save_path, save_name):
        pickle.dump(constructed_task, open(f"{save_path}/{save_name}.pickle", "wb"))


def make_constructed_dataset(dataset, task_num, list_of_class_nums, data_name=None):
    """
    Make a constructed dataset
    :param dataset: the full dataset path
    :param task_num: the task number
    :param list_of_class_nums: a list of class nums needed in subset
    :return: a list of lists for all relevant datapoints in the dataset
    """
    # get task
    constructed = ConstructedDataset(dataset, data_name)
    constructed.change_task(task_num)

    # construct the dataset subset
    cons_data = constructed.create_constructed_data(list_of_class_nums)

    return cons_data, constructed.data_name


def make_multiple_constructed_datasets(
    list_of_datasets,
    list_of_task_nums,
    nested_list_of_class_nums,
    save_path=None,
    list_of_save_names=None,
    data_name=None,
):
    """
    Get multiple constructed datasets
    Needed because train, dev, and test are separate pickle file path/names
    :param list_of_datasets: a list of dataset paths
    :param list_of_task_nums: a list of task_nums/ys_idx
        in the same order as list of datasets
    :param nested_list_of_class_nums: a list of lists of class nums
        in the same order as list of datasets
    :param save_path: path to dir where pickle files will be saved; if None,
        this function returns data
    :param list_of_save_names: list of names to save datasets as
        if None, names are taken from the dataset names + task + class nums
    :return: dict of save_name: data if no save_path is provided
    """
    datasets = {}

    for i, dataset in enumerate(list_of_datasets):
        data_points, data_name = make_constructed_dataset(
            dataset, list_of_task_nums[i], nested_list_of_class_nums[i], data_name
        )

        if list_of_save_names is None:
            if data_name.endswith(".pickle"):
                the_name = data_name.split("/")[-1].split(".pickle")[0]
            else:
                the_name = data_name
            the_task = f"task{str(list_of_task_nums[i])}"
            the_classes = (
                f"classes{'-'.join([str(n) for n in nested_list_of_class_nums[i]])}"
            )
            save_name = f"{the_name}_{the_task}_{the_classes}"
        else:
            save_name = list_of_save_names[i]

        if save_path is not None:
            # save the data points for this task to pickle
            pickle.dump(data_points, open(f"{save_path}/{save_name}.pickle", "wb"))
        else:
            datasets[save_name] = data_points

    if save_path is None:
        return datasets


def make_single_constructed_set_from_multiple_datasets(
    list_of_datasets,
    list_of_task_nums,
    nested_list_of_class_nums,
    save_path,
    save_name=None,
):
    """
    Take multiple datasets and combine relevant subsets of each into a single
    constructed dataset that is saved
    :param list_of_datasets: a list of dataset paths
    :param list_of_task_nums: a list of task_nums/ys_idx
        in the same order as list of datasets
    :param nested_list_of_class_nums: a list of lists of class nums
        in the same order as list of datasets
    :param save_path: path to dir where pickle files will be saved
    :param save_name: names to save dataset as
        if None, name is taken from the dataset names + task + class nums
    :return:
    TODO: currently only accepts input paths and not unpickled datasets
    """
    # holder for all data
    all_data = []

    # counter for classes
    total_classes = 0

    # get names of all datasets
    all_dataset_save_info = []

    for i, dataset in enumerate(list_of_datasets):
        # get save name information
        # todo: this will result in long names
        #   condense eventually
        if save_name is None:
            the_name = dataset.split("/")[-1].split(".pickle")[0]
            the_task = f"{str(list_of_task_nums[i])}"
            the_classes = f"{'-'.join([str(n) for n in nested_list_of_class_nums[i]])}"
            all_dataset_save_info.append(f"{the_name}-t{the_task}-c{the_classes}")

        # get relevant data
        data_points = make_constructed_dataset(
            dataset, list_of_task_nums[i], nested_list_of_class_nums[i]
        )

        # change class numbers for ys data
        for point in data_points:

            # add original task + class num to separate key in dict
            point["original_task_and_class"] = (
                list_of_task_nums[i],
                point["ys"][list_of_task_nums[i]],
            )

            # assumes data point is in dict format
            point["ys"][list_of_task_nums[i]] += total_classes

            # add point to data holder
            all_data.append(point)

        # increment counter of all classes
        total_classes += len(nested_list_of_class_nums[i])

    # shuffle the data
    # uses random seed set at top of this script
    random.shuffle(all_data)

    # set the save name
    if save_name is None:
        save_name = "_".join(all_dataset_save_info)

    # save the data points for this task to pickle
    pickle.dump(all_data, open(f"{save_path}/{save_name}.pickle", "wb"))


if __name__ == "__main__":
    base_path = "../../datasets/pickled_data"

    train = f"{base_path}/mustard_IS13_distilbert_dict_train.pickle"
    dev = f"{base_path}/mustard_IS13_distilbert_dict_dev.pickle"
    test = f"{base_path}/mustard_IS13_distilbert_dict_test.pickle"

    all = [train, dev, test]

    tasks = [0, 0, 0]
    classes = [[1], [1], [1]]

    make_multiple_constructed_datasets(all, tasks, classes, base_path)
    make_single_constructed_set_from_multiple_datasets(all, tasks, classes, base_path)
