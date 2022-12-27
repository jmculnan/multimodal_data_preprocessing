# adapted from https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html

import os
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import pandas as pd

import librosa
import matplotlib.pyplot as plt


def get_speech_sample(path, resample=None):
    effects = [["remix", "1"]]
    if resample:
        effects.extend([
            ["lowpass", f"{resample // 2}"],
            ["rate", f'{resample}'],
        ])
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


def get_and_save_spectrogram(path_to_dir, audiofile, save_dir, resample=None):
    # get full path to file
    path_to_audio = f"{path_to_dir}/{audiofile}"

    # get name of audio without extension
    audioname = audiofile.split('.wav')[0]

    # check to see if save path exists; if not, make it
    os.makedirs(save_dir, exist_ok=True)

    waveform, sample_rate = get_speech_sample(path_to_audio, resample=resample)

    # define transformation
    spectrogram = T.Spectrogram(
        n_fft=1024,
        win_length=None,
        hop_length=512,
        #center=True,
        #pad_mode="reflect",
        power=2.0,
    )

    # Perform transformation
    spec = spectrogram(waveform)
    spec = spec.squeeze(0)
    spec = torch.transpose(spec, 0, 1)

    spec = pd.DataFrame(spec).astype("float")

    # save to csv file
    spec.to_csv(f"{save_dir}/{audioname}.csv", index=False)



def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None, save_name=None):
    fig, axs = plt.subplots(1, 1)
    # added
    plt.grid(True)

    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    # # add losses/epoch for train and dev set to plot
    # ax.plot(epoch, train_vals, label="train")
    # ax.plot(epoch, dev_vals, label="dev")
    # save the file
    if save_name is not None:
        plt.savefig(fname=save_name)
        plt.close()
    else:
        plt.show(block=False)


if __name__ == "__main__":
    the_files = "../../asist_data/sent-emo/for_PI_meeting_07.22/split"
    #the_files = "../../datasets/multimodal_datasets/MELD_formatted/train/train_audio_mono"
    output = "../../asist_data/sent-emo/for_PI_meeting_07.22/spectrograms"

    max_size = None

    for f in os.listdir(the_files):
        if f.endswith(".wav"):
            get_and_save_spectrogram(the_files, f, output, None)
