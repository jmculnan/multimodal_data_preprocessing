from sklearn.model_selection import train_test_split

from prep_data import *
from utils.data_prep_helpers import Glove, make_glove_dict, get_data_samples


def prep_meld_data(
    data_path="../../datasets/multimodal_datasets/meld_formatted",
    feature_set="IS13",
    transcription_type="gold",
    embedding_type="distilbert",
    glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
    features_to_use=None,
    as_dict=False,
    avg_acoustic_data=False,
    custom_feats_file=None,
    num_train_ex=None,
    include_spectrograms=False
):
    # load glove
    if embedding_type.lower() == "glove":
        glove_dict = make_glove_dict(glove_filepath)
        glove = Glove(glove_dict)
    else:
        glove = None

    # holder for name of file containing utterance info
    if transcription_type.lower() == "gold":
        utts_name = "sent_emo.csv"
    else:
        utts_name = f"meld_{transcription_type.lower()}.tsv"

    # create instance of StandardPrep class
    meld_prep = StandardPrep(
        data_type="meld",
        data_path=data_path,
        feature_set=feature_set,
        utterance_fname=utts_name,
        glove=glove,
        transcription_type=transcription_type,
        use_cols=features_to_use,
        avg_acoustic_data=avg_acoustic_data,
        custom_feats_file=custom_feats_file,
        bert_type=embedding_type,
        include_spectrograms=include_spectrograms
    )

    print("Now preparing training data")
    train_data = meld_prep.train_prep.combine_xs_and_ys(as_dict=as_dict)
    print("Now preparing development data")
    dev_data = meld_prep.dev_prep.combine_xs_and_ys(as_dict=as_dict)
    print("Now preparing test data")
    test_data = meld_prep.test_prep.combine_xs_and_ys(as_dict=as_dict)

    # update train and dev
    train_and_dev = train_data + dev_data
    train_data, dev_data = train_test_split(train_and_dev, test_size=0.2, random_state=88)

    # todo: fix weights so they are only coming from repartitioned train
    class_weights = meld_prep.train_prep.class_weights

    if num_train_ex:
        train_data = get_data_samples(train_data, num_train_ex)

    return train_data, dev_data, test_data, class_weights
