from prep_data import *
from utils.data_prep_helpers import (
    Glove,
    make_glove_dict,
    get_speaker_to_index_dict,
    get_data_samples,
)


def prep_mustard_data(
    data_path="../../datasets/multimodal_datasets/mustard",
    feature_set="IS13",
    transcription_type="gold",
    embedding_type="distilbert",
    glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
    features_to_use=None,
    as_dict=False,
    avg_acoustic_data=False,
    custom_feats_file=None,
    num_train_ex=None,
    include_spectrograms=False,
):
    # load glove
    if embedding_type.lower() == "glove":
        glove_dict = make_glove_dict(glove_filepath)
        glove = Glove(glove_dict)
    else:
        glove = None

    # holder for name of file containing utterance info
    if transcription_type.lower() == "gold":
        utts_name = "mustard_utts.tsv"
    else:
        utts_name = f"mustard_{transcription_type.lower()}.tsv"

    # create instance of StandardPrep class
    mustard_prep = SelfSplitPrep(
        data_type="mustard",
        data_path=data_path,
        feature_set=feature_set,
        utterance_fname=utts_name,
        glove=glove,
        use_cols=features_to_use,
        as_dict=as_dict,
        avg_acoustic_data=avg_acoustic_data,
        custom_feats_file=custom_feats_file,
        bert_type=embedding_type,
        include_spectrograms=include_spectrograms
    )

    # get train, dev, test data
    train_data, dev_data, test_data = mustard_prep.get_data_folds()

    # get train ys
    if as_dict:
        train_ys = [int(item["ys"][0]) for item in train_data]
    else:
        train_ys = [int(item[4]) for item in train_data]

    # get updated class weights using train ys
    class_weights = get_updated_class_weights(train_ys)

    if num_train_ex:
        train_data = get_data_samples(train_data, num_train_ex)

    return train_data, dev_data, test_data, class_weights
