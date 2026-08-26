import itertools

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from . import control, prediction, viz

plt.ion()

env = gym.make("CartPole-v1", render_mode=None)

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
    G = 0
    for step in itertools.count():
        action = control.action(env, resolve(o_b))
        observation, reward, terminated, truncated, info = env.step(action)
        o_b = bin(observation)
        G = G * gamma + reward

        # prediction
        if not visited[tuple(o_b)]:
            visited[tuple(o_b)] = True
            v = q[tuple(o_b)]
            q[tuple(o_b)] = (1 / (episode + 1)) * v + (v - G)

        if truncated or terminated:
            break

    print(episode)
    plt.plot(viz.value_by_angle(q))


env.close()

plt.ioff()
plt.show()


def main():
    pass
