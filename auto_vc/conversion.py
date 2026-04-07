"""
Convert spectrograms using trained AutoVC model.

Loads eval.pkl, runs all speaker-pair conversions through AutoVC,
and saves the converted spectrograms to results.pkl.
"""
import os
import pickle
from math import ceil

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.model_vc import Generator


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0]) / base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0, len_pad), (0, 0)), "constant"), len_pad


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="vctk"
)
def main(config: DictConfig):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    G = Generator(
        config.model.dim_neck,
        config.model.dim_emb,
        config.model.dim_pre,
        config.model.freq,
    ).eval().to(device)
    g_checkpoint = torch.load(
        config.path.autovc_checkpoint, map_location=device, weights_only=False
    )
    G.load_state_dict(g_checkpoint["model"])

    metadata = pickle.load(
        open(os.path.join(config.path.target_dir, "eval.pkl"), "rb")
    )

    spect_vc = []

    for sbmt_i in metadata:

        x_org = sbmt_i[2]
        x_org, len_pad = pad_seq(x_org)
        uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
        emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)

        for sbmt_j in metadata:

            emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)

            with torch.no_grad():
                _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)

            if len_pad == 0:
                uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            else:
                uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

            spect_vc.append(("{}x{}".format(sbmt_i[0], sbmt_j[0]), uttr_trg))

    results_path = os.path.join(config.path.target_dir, "results.pkl")
    with open(results_path, "wb") as handle:
        pickle.dump(spect_vc, handle)

    print(f"Saved {results_path} for {len(spect_vc)} conversions")


if __name__ == "__main__":
    main()
