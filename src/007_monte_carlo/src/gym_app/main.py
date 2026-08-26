import itertools

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from . import control, prediction, viz

fig, ax = plt.subplots()
(ln,) = ax.plot([], [])
plt.show(block=False)

env = gym.make("CartPole-v1", render_mode=None)

q = np.zeros(shape=(100,) * 4, dtype=np.float64)
N = np.zeros(shape=(100,) * 4, dtype=np.uint8)

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


def resolve(bin: NDArray[np.uint8]) -> NDArray[np.float64]:
    return bins[np.arange(4), bin]  # todo out of bounds exception


for episode in itertools.count():
    N.fill(False)
    observation, info = env.reset()
    o_b = bin(observation)
    G = 0
    for step in itertools.count():
        action = control.action(env, resolve(o_b))
        observation, reward, terminated, truncated, info = env.step(action)
        o_b = bin(observation)
        G = G * gamma + reward
        N[tuple(o_b)] += 1

        # prediction
        if N[tuple(o_b)] == 1:
            v = q[tuple(o_b)]
            q[tuple(o_b)] += (1 / N[tuple(o_b)]) * (G - v)

        if truncated or terminated:
            break

    print(episode)
    y = viz.value_by_angle(q)
    x = np.arange(len(y))
    ln.set_data(x, y)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)


env.close()


def main():
    pass
