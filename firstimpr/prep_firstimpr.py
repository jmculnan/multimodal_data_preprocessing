from prep_data import *
from utils.data_prep_helpers import Glove, make_glove_dict


def prep_firstimpr_data(
    data_path="../../datasets/multimodal_datasets/chalearn",
    feature_set="IS13",
    transcription_type="gold",
    glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
    features_to_use=None,
    pred_type="max_class",
):
    # load glove
    glove_dict = make_glove_dict(glove_filepath)
    glove = Glove(glove_dict)

    # holder for name of file containing utterance info
    if transcription_type.lower() == "gold":
        utts_name = "gold_and_utts.tsv"
    else:
        utts_name = f"chalearn_{transcription_type.lower()}.tsv"

    # create instance of StandardPrep class
    firstimpr_prep = StandardPrep(
        data_type="firstimpr",
        data_path=data_path,
        feature_set=feature_set,
        utterance_fname=utts_name,
        glove=glove,
        transcription_type=transcription_type,
        use_cols=features_to_use,
    )

    # add the prediction type, since first impressions can have several
    firstimpr_prep.train_prep.add_pred_type(pred_type)
    firstimpr_prep.dev_prep.add_pred_type(pred_type)
    firstimpr_prep.test_prep.add_pred_type(pred_type)

    # get train, dev, test data
    print("Now preparing training data")
    train_data = firstimpr_prep.train_prep.combine_xs_and_ys()
    print("Now preparing development data")
    dev_data = firstimpr_prep.dev_prep.combine_xs_and_ys()
    print("Now preparing test data")
    test_data = firstimpr_prep.test_prep.combine_xs_and_ys()

    # get class weights
    class_weights = firstimpr_prep.train_prep.class_weights

    return train_data, dev_data, test_data, class_weights


if __name__ == "__main__":
    train, dev, test, weights = prep_firstimpr_data()

    print(weights)
    print(type(train))
