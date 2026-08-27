import gymnasium as gym
from numpy.typing import NDArray


# Random policy
def action(env: gym.Env, state, Q: NDArray):
    return env.action_space.sample()
