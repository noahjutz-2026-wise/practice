import os

import ray
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.tune import CheckpointConfig

from hello_torchrl.environment import flappy_bird

WANDB_API_KEY = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")


def main():
    ctx = ray.init(
        address="auto",
        runtime_env={},
    )
    tune.register_env("flappy-bird", lambda _: flappy_bird.env)
    config = (
        DreamerV3Config()
        .environment("flappy-bird")
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=1,
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=1,
        )
        .training(
            model_size="XS",
            training_ratio=2,
            batch_size_B=8,
            batch_length_T=32,
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": 10000,
            },
        )
    )

    results = tune.Tuner(
        trainable="DreamerV3",
        param_space=config,
        run_config=tune.RunConfig(
            storage_path="/var/tmp/ray_results",
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=10, checkpoint_at_end=True, num_to_keep=2
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
