import os
from pathlib import Path

import ray
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.tune import CheckpointConfig

from hello_torchrl.environment import flappy_bird

WANDB_API_KEY = os.environ["WANDB_API_KEY"]
RAY_TEMP_DIR = Path(os.environ["RAY_TEMP_DIR"])


def main():
    ctx = ray.init(
        address="auto",
        runtime_env={},
    )
    tune.register_env("flappy-bird", flappy_bird.get_env)
    config = (
        DreamerV3Config()
        .environment("flappy-bird")
        .env_runners(
            num_env_runners=8,
            num_envs_per_env_runner=4,
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .training(
            model_size="XS",
            training_ratio=1,
            batch_size_B=32,
        )
    )

    results = tune.Tuner(
        trainable="DreamerV3",
        param_space=config,
        run_config=tune.RunConfig(
            storage_path=(RAY_TEMP_DIR / "ray_results").name,
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=1000, checkpoint_at_end=True, num_to_keep=2
            ),
            callbacks=[
                WandbLoggerCallback(
                    project="ray_dreamer",
                    entity="tjno",
                    api_key=WANDB_API_KEY,
                    log_config=True,
                )
            ],
            # stop={"training_iteration": 100},
        ),
    ).fit()


if __name__ == "__main__":
    main()
