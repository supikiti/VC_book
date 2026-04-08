import os

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from random import shuffle


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config"
)
def main(config: DictConfig):
    source_dir = config.path.wav_16k_dir

    os.makedirs(config.path.filelist_dir, exist_ok=True)

    train = []
    val = []
    test = []

    for speaker in tqdm(sorted(os.listdir(source_dir))):
        wavs = os.listdir(os.path.join(source_dir, speaker))
        shuffle(wavs)
        train += [speaker + "/" + name for name in wavs[2:-10]]
        val += [speaker + "/" + name for name in wavs[:2]]
        test += [speaker + "/" + name for name in wavs[-10:]]

    shuffle(train)
    shuffle(val)
    shuffle(test)

    for path, data in [
        (config.path.training_files, train),
        (config.path.validation_files, val),
        (config.path.test_files, test),
    ]:
        print("Writing", path)
        with open(path, "w") as f:
            for fname in tqdm(data):
                f.write(fname + "\n")


if __name__ == "__main__":
    main()
