import os
import pickle

import hydra
import soundfile as sf
import torch
from omegaconf import DictConfig

from src.synthesis import build_model
from src.synthesis import wavegen


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="vctk"
)
def main(config: DictConfig):
    # WaveNet vocoderはMPS非対応のため、CUDAかCPUのみ使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocoder model
    model = build_model().to(device)
    checkpoint = torch.load(
        config.path.vocoder_checkpoint,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])

    # Load converted spectrograms
    results_path = os.path.join(config.path.target_dir, "results.pkl")
    spect_vc = pickle.load(open(results_path, "rb"))

    eval_dir = config.path.eval_dir
    os.makedirs(eval_dir, exist_ok=True)

    for spect in spect_vc:
        name = spect[0]
        c = spect[1]
        print(name)
        waveform = wavegen(model, c=c)
        sf.write(os.path.join(eval_dir, name + ".wav"), waveform, samplerate=16000)


if __name__ == "__main__":
    main()
