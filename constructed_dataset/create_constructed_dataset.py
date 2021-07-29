# create a constructed dataset by combining individual datasets
# add task label to each

from scripts.save_partitioned_data import prep_data
from utils.data_prep_helpers import look_up_num_classes
import pickle


class ConstructedDataset:
    def __init__(self, pickled_dataset):
        """
        :param pickled_dataset: path to a pickled data file
        """
        self.pickled = pickled_dataset
        self.dataset = self.unpickle()
        self.data_is_dict = True if type(self.dataset[0]) == dict else False

        print(self.data_is_dict)

        if self.data_is_dict:
            self.task_num = 0
        else:
            # get ys idx
            self.ys_idx = 4

        self.class_nums = []

    def unpickle(self):
        with open(self.pickled, 'rb') as pfile:
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

    def create_constructed_data(self):
        """
        Select a portion of a dataset consisting of a specific class number
        :return: a dataset composed only of desired classes
        """
        # create a dataset holder
        data_subset = []

        # for item in dataset
        for item in self.dataset:

            # if item[class_num_idx] in class_nums
            if self.data_is_dict:
                # ys is a key, val is a list of lists
                if item["ys"][self.task_num] in self.class_nums:
                    data_subset.append(item)
            else:
                if item[self.ys_idx] in self.class_nums:
                    # add to dataset holder
                    data_subset.append(item)

        # return
        return data_subset

    def save_constructed_task(self, constructed_task, save_path, save_name):
        pickle.dump(
            constructed_task, open(f"{save_path}/{save_name}.pickle", "wb")
        )


def get_multiple_constructed_datasets(list_of_datasets, list_of_task_nums, list_of_class_nums,
                                      save_path, list_of_save_names=None):
    """
    Get multiple constructed datasets
    Needed because train, dev, and test are separate pickle file path/names
    :param list_of_datasets: a list of dataset paths
    :param list_of_task_nums: a list of task_nums/ys_idx
        in the same order as list of datasets
    :param list_of_class_nums: a list of lists of class nums
        in the same order as list of datasets
    :param save_path: path to dir where pickle files will be saved
    :param list_of_save_names: list of names to save datasets as
        if None, names are taken from the dataset names + task + class nums
    :return:
    """
    for i, dataset in enumerate(list_of_datasets):
        constructed = ConstructedDataset(dataset)
        constructed.change_task(list_of_task_nums[i])
        constructed.add_class_nums(list_of_class_nums[i])
        cons_data = constructed.create_constructed_data()

        if list_of_save_names is None:
            the_name = dataset.split('/')[-1].split('.pickle')[0]
            the_task = f"task{str(list_of_task_nums[i])}"
            the_classes = f"classes{'-'.join([str(n) for n in list_of_class_nums[i]])}"
            save_name = f"{the_name}_{the_task}_{the_classes}"
        else:
            save_name = list_of_save_names[i]
        constructed.save_constructed_task(cons_data, save_path, save_name)


#
# def create_constructed_data(
#         data_list,
#         data_paths,
#         feature_set,
#         transcription_type,
#         glove_path,
#         feats_to_use=None,
#         combine_partitions=True,
# ):
#     """
#     Get a constructed dataset containing all data
#     :param data_list: A list of datasets to include
#         if pred type is needed, that item should be
#         a (dataset, predtype) double
#     :param data_paths: list of paths to the datasets
#     :return:
#     """
#     # set a counter for the number of classes
#     classes = 0
#
#     # set holder for combined data
#     combined_data = []
#
#     # get each dataset
#     for i, dset in enumerate(data_list):
#         # get the data folds
#         if type(dset) == str:
#             train, dev, test, clsswt = prep_data(dset, data_paths[i], feature_set, transcription_type, glove_path,
#                                                  feats_to_use, data_as_dict=True)
#             dset_name = dset
#
#         elif type(dset) == tuple:
#             train, dev, test, clsswt = prep_data(dset[0], data_paths[i], feature_set, transcription_type, glove_path,
#                                                  feats_to_use, dset[1], data_as_dict=True)
#
#             dset_name = f"{dset[0]}_{dset[1]}"
#
#         # combine partitions if need be
#         if combine_partitions:
#             all_data = train + dev + test
#
#         # get number of classes in this dataset
#         num_classes = look_up_num_classes(dset_name)
#
#         # alter the y values (class numbers)
#         for item in all_data:
#             # set the dataset number
#             item["dataset_num"] = i
#             # set the class number out of the pool of all classes
#             item["ys"][0] = item["ys"][0] + classes
#             # if num_classes is a list, though, there are 2 y values
#             if type(num_classes) == list:
#                 # add both the number of classes in item 1 + classes counter
#                 item["ys"][1] = item["ys"][1] + classes + num_classes[0]
#
#         # add updated data to all data holder
#         combined_data.extend(all_data)
#
#         # add number of classes seen to class counter
#         if type(num_classes) == list:
#             num_classes = sum(num_classes)
#
#         classes += num_classes
#
#     print(classes)
#     print(combined_data[::10000])
#     # return altered dataset
#     return combined_data


if __name__ == "__main__":
    base_path = "../../datasets/pickled_data"

    train = f"{base_path}/mustard_IS13_distilbert_dict_train.pickle"
    dev = f"{base_path}/mustard_IS13_distilbert_dict_dev.pickle"
    test = f"{base_path}/mustard_IS13_distilbert_dict_test.pickle"

    all = [train, dev, test]

    tasks = [0, 0, 0]
    classes = [[1], [1], [1]]

    get_multiple_constructed_datasets(all, tasks, classes, base_path)
