import h5py
import re

from prep_data import SelfSplitPrep, get_updated_class_weights
from utils.data_prep_helpers import make_glove_dict, Glove, get_data_samples

from utils.audio_extraction import run_feature_extraction


def prep_mosi_data(
    data_path="../../datasets/multimodal_datasets/CMU_MOSI",
    feature_set="IS13",
    transcription_type="gold",
    embedding_type="distilbert",
    glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
    features_to_use=None,
    pred_type="classification",
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
    utts_name = f"mosi_{transcription_type.lower()}.tsv"

    # create instance of StandardPrep class
    mosi_prep = SelfSplitPrep(
        data_type="mosi",
        data_path=data_path,
        feature_set=feature_set,
        utterance_fname=utts_name,
        glove=glove,
        use_cols=features_to_use,
        pred_type=pred_type,
        as_dict=as_dict,
        avg_acoustic_data=avg_acoustic_data,
        custom_feats_file=custom_feats_file,
        bert_type=embedding_type,
        include_spectrograms=include_spectrograms
    )

    # get train, dev, test data
    train_data, dev_data, test_data = mosi_prep.get_data_folds()

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


def convert_gold_labels_to_tsv(
    gold_path, gold_file="CMU_MOSI_Opinion_Labels.csd", save_name="mosi_gold.tsv"
):
    """
    Read in gold labels file with h5py
    Organize gold labels data + read to tsv
    :param gold_path: path of gold .csd
    :param gold_file: name of gold .csd
    :param save_name: name to save the gold file within gold_path
    :return:
    """
    # get holder for output
    all_organized_data = ["speaker\tid\tutterance\ttime_start\ttime_end\tsentiment"]
    # read in file
    all_data = h5py.File(f"{gold_path}/{gold_file}", "r")
    # get data
    data = all_data["Opinion Segment Labels"]["data"]
    # get the names of the files
    names = data.keys()

    for name in names:
        # access corresponding segmented transcript
        transcript_path = f"{gold_path}/Transcript/Segmented/{name}.annotprocessed"
        utt_dict = read_segmented_transcript(transcript_path)

        individual_features_list = data[name]["features"]
        individual_intervals_list = data[name]["intervals"]

        for i, score in enumerate(individual_features_list):
            score = score[0]
            start_time = individual_intervals_list[i][0]
            end_time = individual_intervals_list[i][1]
            id = f"{name}_{str(i+1)}"
            all_organized_data.append(
                f"{name}\t{id}\t{utt_dict[(name, i + 1)]}\t{start_time}\t{end_time}\t{score}"
            )

    with open(f"{gold_path}/{save_name}", "w") as save_file:
        save_file.write("\n".join(all_organized_data))


def read_segmented_transcript(path_to_transcript):
    """
    Read in a segmented transcript and organize into dict
    :return: A dict of (name, num): utterance pairs
    """
    segment_dict = {}

    name = re.search(r"(\S+).annotprocessed", path_to_transcript.split("/")[-1]).group(
        1
    )

    with open(path_to_transcript, "r") as tfile:
        for line in tfile:
            # find segment number
            num = int(re.search(r"([0-9]+)_DELIM_", line).group(1))

            # get utterance
            line = re.sub(r"[0-9]+_DELIM_", "", line)
            line = line.strip()

            # add to dict
            segment_dict[(name, num)] = line

    return segment_dict


if __name__ == "__main__":
    # audio_path = "../../datasets/multimodal_datasets/CMU_MOSI"
    # audio_extension = "Audio/WAV_16000/segmented"
    #
    # run_feature_extraction(f"{audio_path}/{audio_extension}", "IS13", f"{audio_path}/IS13")

    # base_path = "../../datasets/multimodal_datasets/CMU_MOSI"
    # convert_gold_labels_to_tsv(base_path)

    train, dev, test, weights = prep_mosi_data()
    #
    # print(train[0])
    # print(len(train))
    # print(weights)
