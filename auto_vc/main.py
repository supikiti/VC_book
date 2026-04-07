import hydra
from omegaconf import DictConfig
from src.solver_encoder import Solver
from src.data_loader import get_loader
from torch.backends import cudnn


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="vctk"
)
def main(config: DictConfig):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    vcc_loader = get_loader(
        config.model.data_dir, config.model.batch_size, config.model.len_crop
    )

    solver = Solver(vcc_loader, config.model, config.tag)

    solver.train()


if __name__ == "__main__":
    main()
