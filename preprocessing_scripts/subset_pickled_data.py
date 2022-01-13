import pickle
import random


def subset_data(
    pickled_file, num_data_points, equal_classes=True, num_points_eq_k=False
):
    # get the name of the pickled file
    # assumes file ends with extension .pickle or .p
    # if equal_classes, every class has same number of points
    # if num_points_eq_k and equal classes, num_data_points is used for EVERY class
    file_name_parts = pickled_file.split("/")

    if pickled_file.endswith(".p"):
        file_name = file_name_parts[-1][:-2]
        ext = ".p"
    elif pickled_file.endswith(".pickle"):
        file_name = file_name_parts[-1][:-7]
        ext = ".pickle"

    new_name = f"{file_name}_{str(num_data_points)}{ext}"
    save_path = f"{'/'.join(file_name_parts[:-1])}/{new_name}"

    data = pickle.load(open(pickled_file, "rb"))

    if equal_classes:
        new_data = []
        sorted_data = separate_data_by_class(data)
        print(len(sorted_data.keys()))
        for key in sorted_data.keys():
            print(len(sorted_data[key]))
        for y_class in sorted_data.keys():
            print(y_class)
            if num_data_points > len(sorted_data[y_class]):
                print(f"{num_data_points} is greater than {len(sorted_data[y_class])}")
                if not num_points_eq_k:
                    new_data.extend(
                        random.choices(
                            sorted_data[y_class],
                            k=round(num_data_points / len(sorted_data.keys())),
                        )
                    )
                    print(len(new_data))
                else:
                    new_data.extend(
                        random.choices(sorted_data[y_class], k=num_data_points)
                    )
            else:
                print(f"{num_data_points} is less than {len(sorted_data[y_class])}")
                if not num_points_eq_k:
                    new_data.extend(
                        random.sample(
                            sorted_data[y_class],
                            round(num_data_points / len(sorted_data.keys())),
                        )
                    )
                    print(len(new_data))
                else:
                    new_data.extend(
                        random.sample(sorted_data[y_class], num_data_points)
                    )

        # shuffle the data, since it's organized by class
        random.shuffle(new_data)
    else:
        if num_data_points > len(data):
            new_data = random.choices(data, k=num_data_points)
        else:
            new_data = random.sample(data, num_data_points)

    pickle.dump(new_data, open(save_path, "wb"))

    print(f"New data with {num_data_points} data points saved:")
    print(save_path)


def separate_data_by_class(dataset):
    """
    Separate the data in a dataset by y class
    :param dataset: a list of data points; each item is a dict
    :return: a dict of lists of data points
    """
    data_by_class = {}

    # get the set of y values
    class_nums = set([int(item["ys"][0]) for item in dataset])

    # add each y value to holder
    for num in class_nums:
        data_by_class[num] = []

    # sort data
    for item in dataset:
        data_by_class[int(item["ys"][0])].append(item)

    return data_by_class


if __name__ == "__main__":
    num_data_points = 1000

    base = "../../datasets/pickled_data/IS13_Matching_GOLD"

    meld = f"{base}/meld_IS13_distilbert_dict_train.pickle"
    cdc = f"{base}/cdc_IS13_distilbert_dict_train.pickle"
    firstimpr = f"{base}/firstimpr_IS13_distilbert_dict_train.pickle"
    mosi = f"{base}/mosi_IS13_distilbert_dict_train.pickle"
    rav = f"{base}/ravdess_IS13_distilbert_dict_train.pickle"

    all_data = [meld, cdc, firstimpr, mosi, rav]

    for item in all_data:
        subset_data(item, num_data_points, equal_classes=True, num_points_eq_k=True)
