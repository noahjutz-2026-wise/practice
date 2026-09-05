import os

import ray
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config

WANDB_API_KEY = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")


def _env_creator(ctx):
    import gymnasium as gym
    from ray.rllib.env.wrappers.atari_wrappers import NormalizedImageEnv
    from supersuit.generic_wrappers import resize_v1

    from hello_torchrl.flappy_bird import (  # pyright: ignore[reportMissingTypeStubs]
        flappy_bird_gymnasium,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    )

    env = gym.make("FlappyBird-v0", render_mode="rgb_array", audio_on=False)
    env = gym.wrappers.AddRenderObservation(env)
    # env = gym.wrappers.HumanRendering(env)
    return NormalizedImageEnv(
        resize_v1(  # resize to 64x64 and normalize images
            env, x_size=64, y_size=64
        )
    )


def main():
    ctx = ray.init(
        address="auto",
        runtime_env={},
    )
    tune.register_env("flappy-bird", _env_creator)
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
            training_ratio=64,
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
            callbacks=[
                WandbLoggerCallback(
                    project="ray_dreamer",
                    entity="tjno",
                    api_key=WANDB_API_KEY,
                    log_config=True,
                )
            ],
            stop={"training_iteration": 100},
        ),
    ).fit()


if __name__ == "__main__":
    main()
