from pathlib import Path

import ray
from ray import tune
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config


def _env_creator(ctx):
    import flappy_bird_gymnasium  # noqa: F401
    import gymnasium as gym
    from ray.rllib.env.wrappers.atari_wrappers import NormalizedImageEnv
    from supersuit.generic_wrappers import resize_v1

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
        object_store_memory=1024 * 1024 * 1024,  # Cap Plasma object store at 1GB
        runtime_env={},
        dashboard_host="0.0.0.0",
    )

    print("DASHBOARD URL:")
    print(ctx.dashboard_url)

    # Register the FlappyBird-rgb-v0 env including necessary wrappers via the
    # `tune.register_env()` API.
    tune.register_env("flappy-bird", _env_creator)

    # Define the `config` variable to use for training.
    config = (
        DreamerV3Config()
        # set the env to the pre-registered string
        .environment("flappy-bird")
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=1,
        )
        .learners(
            num_learners=0,
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

    # Run the tuner job with a 1-iteration limit for testing.
    results = tune.Tuner(
        trainable="DreamerV3",
        param_space=config,
        # run_config=tune.RunConfig(stop={"training_iteration": 1}),
    ).fit()
    ray.shutdown()
    return 0


if __name__ == "__main__":
    main()
