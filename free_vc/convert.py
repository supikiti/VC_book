import os
import time

import hydra
import torch
import librosa
from scipy.io.wavfile import write
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from src import utils
from src.models import SynthesizerTrn
from src.mel_processing import mel_spectrogram_torch
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging
logging.getLogger('numba').setLevel(logging.WARNING)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config"
)
def main(config: DictConfig):
    ptfile = config.path.trained_ckpt
    txtpath = config.path.convert_txt
    outdir = config.path.eval_dir
    spk_encoder_ckpt = config.path.spk_encoder_ckpt

    # Build HParams for model compatibility
    config_dict = OmegaConf.to_container(config, resolve=True)
    hps = utils.HParams(
        train=config_dict["train"],
        data=config_dict["data"],
        model=config_dict["model"],
    )

    os.makedirs(outdir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(ptfile, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(device)

    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder(spk_encoder_ckpt)

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(txtpath, "r") as f:
        for rawline in f.readlines():
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            # tgt
            wav_tgt, _ = librosa.load(
                tgt, sr=hps.data.sampling_rate
            )
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            if hps.model.use_spk:
                g_tgt = smodel.embed_utterance(wav_tgt)
                g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)
            else:
                wav_tgt = torch.from_numpy(
                    wav_tgt
                ).unsqueeze(0).to(device)
                mel_tgt = mel_spectrogram_torch(
                    wav_tgt,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )
            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(device)
            c = utils.get_content(cmodel, wav_src)

            if hps.model.use_spk:
                audio = net_g.infer(c, g=g_tgt)
            else:
                audio = net_g.infer(c, mel=mel_tgt)
            audio = audio[0][0].data.cpu().float().numpy()
            write(os.path.join(
                outdir, f"{title}.wav"
            ), hps.data.sampling_rate, audio)


if __name__ == "__main__":
    main()
