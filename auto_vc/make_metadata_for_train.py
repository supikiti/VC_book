"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from collections import OrderedDict

import hydra
from src.model_bl import D_VECTOR
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


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
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
    c_checkpoint = torch.load(
        config.path.d_vector_checkpoint, map_location='cpu'
    )
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)
    num_uttrs = 10
    len_crop = 128

    # Directory containing mel-spectrograms
    targetDir = config.path.target_dir
    dirName, subdirList, _ = next(os.walk(targetDir))
    print('Found directory: %s' % dirName)

    speakers = []
    for speaker in sorted(subdirList):
        print('Processing speaker: %s' % speaker)
        utterances = []
        utterances.append(speaker)
        _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))

        # make speaker embedding
        assert len(fileList) >= num_uttrs
        idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
        embs = []
        for i in range(num_uttrs):
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
            candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
            # choose another utterance if the current one is too short
            while tmp.shape[0] < len_crop:
                idx_alt = np.random.choice(candidates)
                tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
                candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
            left = np.random.randint(0, tmp.shape[0]-len_crop)
            melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).to(device)
            emb = C(melsp)
            embs.append(emb.detach().squeeze().cpu().numpy())
        utterances.append(np.mean(embs, axis=0))

        # create file list
        for fileName in sorted(fileList):
            utterances.append(os.path.join(speaker,fileName))
        speakers.append(utterances)

    with open(os.path.join(targetDir, 'train.pkl'), 'wb') as handle:
        pickle.dump(speakers, handle)


if __name__ == "__main__":
    main()