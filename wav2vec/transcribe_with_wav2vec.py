# get wav2vec speech representation for input
# adapted from https://maelfabien.github.io/machinelearning/wav2vec/#5-the-code

import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer  # , Wav2Vec2FeatureExtractor
import nltk
import os

def load_model():
    tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return tokenizer, model


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

    return correct_sentence(transcription.lower())

def transcribe_audio_dir(input_dir):
    tokenizer, model = load_model()

    for f in os.listdir(input_dir):
        if f.endswith(".wav"):
            print(f)

            speech, fs = sf.read(f"{input_dir}/{f}")

            if len(speech.shape) > 1:
                speech = speech[:, 0] + speech[:, 1]

            if fs != 16000:
                speech = librosa.resample(speech, fs, 16000)

            input_values = tokenizer(speech, return_tensors="pt", sampling_rate=16000).input_values
            logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)

            transcription = tokenizer.decode(predicted_ids[0])

            print(correct_sentence(transcription.lower()))


if __name__ == "__main__":
    the_files = "../../datasets/multimodal_datasets/MELD_formatted/train/train_audio_mono"

    transcribe_audio_dir(the_files)
