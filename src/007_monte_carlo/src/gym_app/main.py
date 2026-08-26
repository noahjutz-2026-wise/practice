import itertools

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from . import control, prediction

env = gym.make("CartPole-v1", render_mode="human")

q = np.zeros(shape=(100,) * 4, dtype=np.uint8)
visited = np.zeros(shape=(100,) * 4, dtype=np.bool)

bins = np.vstack(
    (
        np.linspace(-4.8, 4.8, num=100),
        np.linspace(-5, 5, num=100),
        np.linspace(-0.418, 0.418, num=100),
        np.linspace(-5, 5, num=100),
    )
)

gamma = 0.9


def bin(observation: NDArray[np.float64]) -> NDArray[np.uint8]:
    return (observation[:, None] >= bins).sum(axis=1)


def resolve(bin: NDArray[np.uint8]):
    return bins[np.arange(4), bin]


for episode in range(100):
    visited.fill(False)
    observation, info = env.reset()
    o_b = bin(observation)
    total_reward = 0
    for step in itertools.count():
        action = control.action(env, resolve(o_b))
        observation, reward, terminated, truncated, info = env.step(action)
        o_b = bin(observation)

        # prediction
        if not visited[tuple(o_b)]:
            visited[tuple(o_b)] = True
            q[tuple(o_b)] = gamma * q[tuple(o_b)] + reward

        total_reward += reward
        if truncated or terminated:
            break

    prediction.evaluate(q, ...)


env.close()

print(total_reward)


def main():
    pass
