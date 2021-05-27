import argparse
import pickle

from prep_data import *
from combine_xs_and_ys_by_dataset import combine_xs_and_ys_meld
from make_data_tensors_by_dataset import make_data_tensors_meld

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_path",
    help="Path to dataset",
    default="../../datasets/multimodal_datasets/meld_formatted",
    nargs="?"
)
parser.add_argument(
    "feature_set",
    help="Which feature set to use",
    default="IS13",
    nargs="?"
)
parser.add_argument(
    "transcription_type",
    help="Transcription type used (gold, google, kaldi, sphinx)",
    default="gold",
    nargs="?"
)
parser.add_argument(
    "glove_filepath",
    help="Path to and file name of pickled glove file",
    default="../asist-speech/data/glove.300d.short.punct.pickle",
    nargs="?"
)
parser.add_argument(
    "features_to_use",
    help="Include a list of acoustic features to be used or leave as 'none' to get all",
    default=None,
    nargs="+"
)

args = parser.parse_args()

# load glove
glove = pickle.load(args.glove_filepath)

# holder for name of file containing utterance info
if args.transcription_type.lower() == "gold":
    utts_name = "sent_emo.csv"
else:
    utts_name = f"meld_{args.transcription_type.lower()}.tsv"

# create instance of StandardPrep class
meld_prep = StandardPrep(
    data_type="meld",
    data_path=args.data_path,
    feature_set=args.feature_set,
    utterance_fname=utts_name,
    glove=glove,
    use_cols=args.features_to_use
)

train_data = meld_prep.train_prep.combine_xs_and_ys()
dev_data = meld_prep.dev_prep.combine_xs_and_ys()
test_data = meld_prep.test_prep.combine_xs_and_ys()

class_weights = meld_prep.train_prep.class_weights
