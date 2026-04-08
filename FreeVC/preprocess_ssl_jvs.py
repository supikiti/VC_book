import os

import hydra
import torch
import librosa
from glob import glob
from tqdm import tqdm
from omegaconf import DictConfig

from src import utils
from wavlm import WavLM, WavLMConfig


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config"
)
def main(config: DictConfig):
    in_dir = config.path.wav_16k_dir
    out_dir = config.path.wavlm_dir
    sr = config.data.sampling_rate
    wavlm_ckpt = config.path.wavlm_ckpt

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading WavLM for content...")
    checkpoint = torch.load(wavlm_ckpt, map_location=device)
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).to(device)
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()
    print("Loaded WavLM.")

    filenames = glob(f'{in_dir}/*/*.wav', recursive=True)

    for filename in tqdm(filenames):
        basename = os.path.basename(filename)
        speaker = filename.split(os.sep)[-2]
        save_dir = os.path.join(out_dir, speaker)
        os.makedirs(save_dir, exist_ok=True)
        wav, _ = librosa.load(filename, sr=sr)
        wav = torch.from_numpy(wav).unsqueeze(0).to(device)
        c = utils.get_content(cmodel, wav)
        save_name = os.path.join(save_dir, basename.replace(".wav", ".pt"))
        torch.save(c.cpu(), save_name)


if __name__ == "__main__":
    main()
