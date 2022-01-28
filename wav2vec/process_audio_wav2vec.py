# get wav2vec speech representation for input
# adapted from https://maelfabien.github.io/machinelearning/wav2vec/#5-the-code

import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2FeatureExtractor, Wav2Vec2Config
import nltk
import os

def load_model():
    # tokenizer = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
    # tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    # model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return tokenizer, model

def extract_feats(input_audio):

    speech, fs = sf.read(input_audio)

    if len(speech.shape) > 1:
        speech = speech[:, 0] + speech[:, 1]

    if fs != 16000:
        speech = librosa.resample(speech, fs, 16000)

    # feats = extractor(speech, return_tensors="pt", sampling_rate=16000)
    # return_attention_mask=True with wav2vec2-large-960h-lv60-self
    #   but False with wav2vec2-base-960h
    feats = tokenizer(speech, return_tensors="pt", sampling_rate=16000, return_attention_mask=True,
                      do_normalize=True).input_values # feature_size=30

    with torch.no_grad():
        # get the last hidden layer as an multidimensional embedding of the sequence
        # todo: will need to aggregate or pad when making model-usable dataset
        feats = model(feats, output_hidden_states=True).hidden_states[-1]

    return feats


def correct_sentence(input_text):
    sentences = nltk.sent_tokenize(input_text)
    return ' '.join([s.replace(s[0], s[0].capitalize(), 1) for s in sentences])


def asr_transcript(input_file):
    tokenizer, model = load_model()

    speech, fs = sf.read(input_file)

    if len(speech.shape) > 1:
        speech = speech[:, 0] + speech[:, 1]

    if fs != 16000:
        speech = librosa.resample(speech, fs, 16000)

    input_values = tokenizer(speech, return_tensors="pt").input_values
    logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])

    # transcription = tokenizer.batch_decode(logits.detach().numpy()).text

    return correct_sentence(transcription.lower())


def transcript_of_dir(input_dir):
    tokenizer, model = load_model()

    item2transcript = {}

    for f in os.listdir(input_dir):
        if f.endswith(".wav"):
            item = f

            speech, fs = sf.read(f"{input_dir}/{f}")

            if len(speech.shape) > 1:
                speech = speech[:, 0] + speech[:, 1]

            if fs != 16000:
                speech = librosa.resample(speech, fs, 16000)

            input_values = tokenizer(speech, return_tensors="pt", sampling_rate=16000,
                                     feature_size=30, do_normalize=True).input_values
            output = model(input_values)
            print(output)
            print(output.logits.shape)
            exit()

            # print(logits.shape)

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = tokenizer.decode(predicted_ids[0])

            # transcription = tokenizer.decode(logits.detach().numpy()).text

            item2transcript[item] = transcription

    return item2transcript


if __name__ == "__main__":
    the_files = "../../datasets/test_then_delete/MELD_formatted/train/train_audio_mono"

    tokenizer, model = load_model()
    # transcript_of_dir(the_files)

    for f in os.listdir(the_files):
        if f.endswith(".wav"):
            item = f
    # i2t = transcript_of_dir(the_files)
            print(extract_feats(f"{the_files}/{f}"))

    # with open("../../datasets/test_then_delete/meld_test_trans.tsv", 'a') as pf:
    #     for k in i2t.keys():
    #         print(f"{k}:\t{i2t[k]}")
    #         #pf.write(f"wav2vec2processor-lg\twav2vec2forctc-lg\t{k}\t{i2t[k]}\n")
