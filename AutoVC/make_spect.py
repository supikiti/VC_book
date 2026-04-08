import argparse
import os
import pickle

import numpy as np
import hydra
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState
from omegaconf import DictConfig


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length//2), mode='reflect')
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(
        x, shape=shape,
        strides=strides
    )
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    return np.abs(result)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="vctk"
)
def main(config: DictConfig):
    rootDir = config.path.root_dir
    targetDir = config.path.target_dir

    mel_basis = mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    dirName, subdirList, _ = next(os.walk(rootDir))
    print('Found directory: %s' % dirName)

    # JVSコーパスのwavサブディレクトリ
    jvs_wav_subdirs = ["parallel100/wav24kHz16bit", "nonpara30/wav24kHz16bit"]

    for subdir in sorted(subdirList):
        print(subdir)
        if not os.path.exists(os.path.join(targetDir, subdir)):
            os.makedirs(os.path.join(targetDir, subdir))

        # JVSの場合はparallel100とnonpara30からwavを収集
        if config.tag == "jvs":
            fileList = []
            for wav_subdir in jvs_wav_subdirs:
                wav_dir = os.path.join(dirName, subdir, wav_subdir)
                if os.path.exists(wav_dir):
                    fileList.extend(
                        [(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith(".wav")]
                    )
        else:
            _, _, files = next(os.walk(os.path.join(dirName, subdir)))
            fileList = [(os.path.join(dirName, subdir), f) for f in files]

        # ディレクトリ名から数値部分を抽出 (e.g. "p225" -> 225, "jvs001" -> 1)
        num_str = ''.join(filter(str.isdigit, subdir))
        prng = RandomState(int(num_str))
        for wav_dir, fileName in sorted(fileList, key=lambda x: x[1]):
            # Read audio file
            x, _ = sf.read(os.path.join(wav_dir, fileName))
            # Remove drifting noise
            y = signal.filtfilt(b, a, x)
            # Ddd a little random noise for model roubstness
            wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
            # Compute spect
            D = pySTFT(wav).T
            # Convert to mel and normalize
            D_mel = np.dot(D, mel_basis)
            D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
            S = np.clip((D_db + 100) / 100, 0, 1)    
            # save spect    
            np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                    S.astype(np.float32), allow_pickle=False)    


if __name__ == "__main__":
    main()