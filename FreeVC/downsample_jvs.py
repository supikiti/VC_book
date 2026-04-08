import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

if not hasattr(np, "complex"):
    np.complex = complex  # numpy >=1.24 compatibility for librosa 0.8.x


def process(config: DictConfig, task):
    speaker, subset, wav_name = task

    wav_path = os.path.join(
        config.path.corpus_dir,
        speaker,
        subset,
        config.preprocess.downsample.wav_subdir,
        wav_name,
    )

    if not os.path.exists(wav_path):
        return

    sr1 = config.preprocess.downsample.sr1
    sr2 = config.preprocess.downsample.sr2

    out_dir1 = os.path.join(config.path.wav_16k_dir, speaker)
    out_dir2 = os.path.join(config.path.wav_22k_dir, speaker)
    os.makedirs(out_dir1, exist_ok=True)
    os.makedirs(out_dir2, exist_ok=True)

    wav, sr = librosa.load(wav_path, sr=None)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav = 0.98 * wav / peak

    wav1 = librosa.resample(wav, orig_sr=sr, target_sr=sr1)
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=sr2)

    save_name = wav_name
    save_path1 = os.path.join(out_dir1, save_name)
    save_path2 = os.path.join(out_dir2, save_name)

    wavfile.write(
        save_path1,
        sr1,
        (wav1 * np.iinfo(np.int16).max).astype(np.int16),
    )
    wavfile.write(
        save_path2,
        sr2,
        (wav2 * np.iinfo(np.int16).max).astype(np.int16),
    )


def collect_tasks(config: DictConfig):
    tasks = []
    ds_config = config.preprocess.downsample
    subsets = list(ds_config.subsets) if ds_config.subsets else []
    extensions = tuple(ds_config.extensions)

    for speaker in sorted(os.listdir(config.path.corpus_dir)):
        speaker_dir = os.path.join(config.path.corpus_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue

        for subset in subsets:
            audio_dir = os.path.join(speaker_dir, subset, ds_config.wav_subdir)
            if not os.path.isdir(audio_dir):
                continue

            for wav_name in os.listdir(audio_dir):
                if wav_name.lower().endswith(extensions):
                    tasks.append((speaker, subset, wav_name))

    return tasks


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    tasks = collect_tasks(config)
    if not tasks:
        raise RuntimeError(
            "No audio files found. Check path.corpus_dir, "
            "preprocess.downsample.subsets, and "
            "preprocess.downsample.wav_subdir in the config."
        )

    for task in tqdm(tasks, desc="Downsampling", unit="file"):
        process(config, task)


if __name__ == "__main__":
    main()
