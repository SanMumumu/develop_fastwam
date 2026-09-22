import hydra
from omegaconf import DictConfig

from fastwam.routed_runtime import run_routed_training
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_routed_training(cfg)


if __name__ == "__main__":
    main()
