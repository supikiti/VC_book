import os
import sys
from pathlib import Path
from functools import partial
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

import glob
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from speaker_encoder.voice_encoder import SpeakerEncoder
from speaker_encoder.audio import preprocess_wav


def build_from_path(in_dir, out_dir, weights_fpath, num_workers=1):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    wavfile_paths = sorted(glob.glob(os.path.join(in_dir, '*.wav')))
    for wav_path in wavfile_paths:
        futures.append(executor.submit(
            partial(_compute_spkEmbed, out_dir, wav_path, weights_fpath)))
    return [future.result() for future in tqdm(futures)]


def _compute_spkEmbed(out_dir, wav_path, weights_fpath):
    utt_id = os.path.basename(wav_path).rstrip(".wav")
    fpath = Path(wav_path)
    wav = preprocess_wav(fpath)

    encoder = SpeakerEncoder(weights_fpath)
    embed = encoder.embed_utterance(wav)
    fname_save = os.path.join(out_dir, f"{utt_id}.npy")
    np.save(fname_save, embed, allow_pickle=False)
    return os.path.basename(fname_save)


def preprocess(in_dir, out_dir_root, spk, weights_fpath, num_workers):
    out_dir = os.path.join(out_dir_root, spk)
    os.makedirs(out_dir, exist_ok=True)
    build_from_path(in_dir, out_dir, weights_fpath, num_workers)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config"
)
def main(config: DictConfig):
    in_dir = config.path.wav_16k_dir
    spk_embed_out_dir = config.path.spk_dir
    spk_encoder_ckpt = config.path.spk_encoder_ckpt
    num_workers = cpu_count()

    print("Number of workers: ", num_workers)
    print("[INFO] spk_embed_out_dir: ", spk_embed_out_dir)
    os.makedirs(spk_embed_out_dir, exist_ok=True)

    sub_folder_list = sorted(os.listdir(in_dir))

    for spk in sub_folder_list:
        print("Preprocessing {} ...".format(spk))
        spk_in_dir = os.path.join(in_dir, spk)
        if not os.path.isdir(spk_in_dir):
            continue
        preprocess(spk_in_dir, spk_embed_out_dir, spk, spk_encoder_ckpt, num_workers)

    print("DONE!")
    sys.exit(0)


if __name__ == "__main__":
    main()
