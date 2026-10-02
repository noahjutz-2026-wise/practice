from pathlib import Path

import gymnasium as gym
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

BIN_NAME = Path("unitybuild") / "unitybuild.x86_64"


def mm_gymnasium(bin_path: str) -> gym.Env:
    env = UnityEnvironment(bin_path)
    return UnityToGymWrapper(env)
