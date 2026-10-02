import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from ..gym_env.unity import make_unity_environment


def run() -> None:
    _ = ray.init(address="auto")

    ppo = PPOConfig().environment(make_unity_environment())

    tune.Tuner(
        "PPO",
        run_config=tune.RunConfig(stop={"training_iteration": 1}),
        param_space=ppo,
    ).fit()
