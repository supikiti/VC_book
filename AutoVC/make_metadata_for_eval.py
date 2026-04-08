"""
Generate speaker embeddings and metadata for evaluation/conversion.

Produces eval.pkl with format:
    [[speaker_name, speaker_embedding(256,), spectrogram(T,80)], ...]
"""
import os
import pickle
from collections import OrderedDict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.model_bl import D_VECTOR


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="vctk"
)
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load speaker encoder (d-vector)
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
    c_checkpoint = torch.load(
        "model/3000000-BL.ckpt", map_location=device, weights_only=False
    )
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint["model_b"].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)

    len_crop = 128
    spmel_dir = config.path.target_dir

    dirName, subdirList, _ = next(os.walk(spmel_dir))
    print("Found directory: %s" % dirName)

    metadata = []
    for speaker in sorted(subdirList):
        print("Processing speaker: %s" % speaker)
        _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))
        fileList = sorted(fileList)

        # Compute speaker embedding (average over all utterances)
        embs = []
        for fileName in fileList:
            tmp = np.load(os.path.join(dirName, speaker, fileName))
            if tmp.shape[0] < len_crop:
                len_pad = len_crop - tmp.shape[0]
                tmp = np.pad(tmp, ((0, len_pad), (0, 0)), "constant")
            left = np.random.randint(0, max(1, tmp.shape[0] - len_crop))
            melsp = torch.from_numpy(
                tmp[np.newaxis, left : left + len_crop, :]
            ).to(device)
            emb = C(melsp)
            embs.append(emb.detach().squeeze().cpu().numpy())
        speaker_emb = np.mean(embs, axis=0)

        # Pick the first spectrogram for conversion
        spect = np.load(os.path.join(dirName, speaker, fileList[0]))

        metadata.append([speaker, speaker_emb, spect])

    with open(os.path.join(config.path.target_dir, "eval.pkl"), "wb") as handle:
        pickle.dump(metadata, handle)

    print(f"Saved eval.pkl for {len(metadata)} speakers")


if __name__ == "__main__":
    main()
