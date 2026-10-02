from pathlib import Path

import gymnasium as gym
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

BIN_NAME = Path("unitybuild") / "unitybuild.x86_64"


def make_unity_environment(bin_path: str = str(BIN_NAME)) -> gym.Env:
    env = UnityEnvironment(bin_path)
    return UnityToGymWrapper(env)
