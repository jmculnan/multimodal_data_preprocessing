# adapted from https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html

import os
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import librosa
import matplotlib.pyplot as plt


def get_speech_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.extend([
      ["lowpass", f"{resample // 2}"],
      ["rate", f'{resample}'],
    ])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)


if __name__ == "__main__":
    the_files = "../../datasets/test_then_delete/MELD_formatted/train/train_audio_mono"

    for f in os.listdir(the_files):
        if f.endswith(".wav"):
            item = f

            waveform, sample_rate = get_speech_sample(f"{the_files}/{f}")

            n_fft = 1024
            win_length = None
            hop_length = 512

            # define transformation
            spectrogram = T.Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
            )

            # Perform transformation
            spec = spectrogram(waveform)

            # print(spec)
            print(spec.shape)
            # plot_spectrogram(spec[0], title='torchaudio')


