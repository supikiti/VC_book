import os
import json

import hydra
import torch
import librosa
from glob import glob
from tqdm import tqdm
from scipy.io import wavfile
from omegaconf import DictConfig

from src import utils
from src.mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
import logging
logging.getLogger('numba').setLevel(logging.WARNING)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config"
)
def main(config: DictConfig):
    in_dir = config.path.wav_22k_dir
    wav_dir = config.path.sr_wav_dir
    ssl_dir = config.path.sr_wavlm_dir
    wavlm_ckpt = config.path.wavlm_ckpt
    sr = config.data.sampling_rate
    sr_min = config.preprocess.sr_augmentation.min
    sr_max = config.preprocess.sr_augmentation.max

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading WavLM for content...")
    checkpoint = torch.load(wavlm_ckpt, map_location=device)
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).to(device)
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()
    print("Loaded WavLM.")

    print("Loading vocoder...")
    vocoder = utils.get_vocoder(0)
    vocoder.eval()
    print("Loaded vocoder.")

    config_path = config.path.hifigan_config
    with open(config_path, "r") as f:
        data = f.read()
    hifigan_config = json.loads(data)
    hps = utils.HParams(**hifigan_config)

    filenames = glob(f'{in_dir}/*/*.wav', recursive=True)

    for filename in tqdm(filenames):
        basename = os.path.basename(filename)
        speaker = filename.split(os.sep)[-2]
        spk_wav_dir = os.path.join(wav_dir, speaker)
        spk_ssl_dir = os.path.join(ssl_dir, speaker)
        os.makedirs(spk_wav_dir, exist_ok=True)
        os.makedirs(spk_ssl_dir, exist_ok=True)
        wav, _ = librosa.load(filename, sr=hps.sampling_rate)
        wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(device)
        mel = mel_spectrogram_torch(
            wav_tensor,
            hps.n_fft,
            hps.num_mels,
            hps.sampling_rate,
            hps.hop_size,
            hps.win_size,
            hps.fmin,
            hps.fmax
        )

        for i in range(sr_min, sr_max + 1):
            ssl_path = os.path.join(spk_ssl_dir, basename.replace(".wav", f"_{i}.pt"))
            wav_path = os.path.join(spk_wav_dir, basename.replace(".wav", f"_{i}.wav"))

            if os.path.exists(ssl_path):
                print(f"{ssl_path} exists. skip.")
                continue

            if os.path.exists(wav_path):
                print(f"{wav_path} exists. skip.")
                continue

            mel_rs = utils.transform(mel, i)
            wav_rs = vocoder(mel_rs)[0][0].detach().cpu().numpy()
            _wav_rs = librosa.resample(wav_rs, orig_sr=hps.sampling_rate, target_sr=sr)
            wav_rs_tensor = torch.from_numpy(_wav_rs).to(device).unsqueeze(0)
            c = utils.get_content(cmodel, wav_rs_tensor)
            torch.save(c.cpu(), ssl_path)
            wavfile.write(
                wav_path,
                sr,
                _wav_rs
            )


if __name__ == "__main__":
    main()
