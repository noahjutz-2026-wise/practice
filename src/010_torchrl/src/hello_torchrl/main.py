from pathlib import Path

import ray
from ray import tune
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config


def _env_creator(ctx):
    import flappy_bird_gymnasium  # doctest: +SKIP
    import gymnasium as gym
    from ray.rllib.env.wrappers.atari_wrappers import NormalizedImageEnv
    from supersuit.generic_wrappers import resize_v1

    base_env = gym.make("FlappyBird-v0", render_mode="rgb_array", audio_on=False)
    pixel_env = gym.wrappers.AddRenderObservation(base_env)
    human_env = gym.wrappers.HumanRendering(pixel_env)
    return NormalizedImageEnv(
        resize_v1(  # resize to 64x64 and normalize images
            human_env, x_size=64, y_size=64
        )
    )


def main():
    ray_tmp = Path.home() / ".ray_tmp"
    ray_tmp.mkdir(parents=True, exist_ok=True)
    ctx = ray.init(_temp_dir=str(ray_tmp))

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
        # play around with the insanely high number of hyperparameters for DreamerV3 ;)
        .training(
            model_size="S",
            training_ratio=1024,
        )
    )

    # Run the tuner job with a 1-iteration limit for testing.
    results = tune.Tuner(
        trainable="DreamerV3",
        param_space=config,
        run_config=tune.RunConfig(stop={"training_iteration": 1}),
    ).fit()
    ray.shutdown()
    return 0


if __name__ == "__main__":
    main()
