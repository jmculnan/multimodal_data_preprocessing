# combine pickle files
# used when dataset is too large and needs to be chunked
from pathlib import Path
import re
import pickle
import torch
import sys
sys.path.append("/home/jculnan/github/multimodal_data_preprocessing")
from sklearn.utils import compute_class_weight
import numpy as np


#from prep_data import get_updated_class_weights


def get_class_weights_for_multiple_tasks(train_ys):
    """
    Get updated class weights
    Because DataPrep assumes you only enter train set
    :return:
    """
    items_of_interest = [item['ys'] for item in train_ys]
    num_preds = len(items_of_interest[0])

    all_weights = []

    for i in range(num_preds):
        labels = [int(y[i]) for y in items_of_interest]
        classes = sorted(list(set(labels)))

        weights = compute_class_weight("balanced", classes=classes, y=labels)

        # add weight for missing personality trait
        # since one personality trait is never predicted
        if len(range(max(labels))) == len(classes):
            weights = weights - 0.001
            weights = np.insert(weights, 1, 100.004) # high weight = infrequent

        these_weights = torch.tensor(weights, dtype=torch.float)
        all_weights.append(these_weights)

    return all_weights


class DataFileCombiner:
    """
    Combine and save pickle files for a dataset
    This expects (at a minimum) ys files
    Assumes train, dev, and test all combined
    Specify what else to include
    """
    def __init__(self, data_location,
                 name1,
                 name2,
                 include_text=True,
                 include_acoustic=True,
                 include_spec=False):
        self.name1 = name1
        self.name2 = name2

        self.path = Path(data_location)

        self.to_process = self._id_files_to_process()

    def _id_files_to_process(self):
        all_to_process = []
        for f in self.path.iterdir():
            if self.name1 == f.name.split("_")[0]:
                name2 = re.sub(self.name1, self.name2, f.name)
                all_to_process.append((f.name, name2))

        return all_to_process

    def process_files(self, save_name):
        for pair in self.to_process:
            f1 = self.path / pair[0]
            p1 = pickle.load(open(f1, 'rb'))
            f2 = self.path / pair[1]
            p2 = pickle.load(open(f2, 'rb'))

            name_end = "_".join(pair[0].split("_")[1:])

            combined = combine_two_files(p1, p2)

            if name_end == "ys_train.pickle":
                clsswts = get_class_weights_for_multiple_tasks(combined)
                print(clsswts)
                with open(self.path / f"{save_name}_clsswts.pickle", 'wb') as pf:
                    pickle.dump(clsswts, pf)
                exit()

            #with open(self.path / f"{save_name}_{name_end}", 'wb') as pf:
            #    pickle.dump(combined, pf)



def combine_two_files(pf1, pf2):
    """
    Combine the information in two pf files
    Order should be preserved
    """
    pf1.extend(pf2)
    return pf1


if __name__ == "__main__":
    base = "/home/jculnan/github/tomcat-speech/data/asist_combine"

    combi = DataFileCombiner(base, "asist", "asist1")

    combi.process_files("ASIST")