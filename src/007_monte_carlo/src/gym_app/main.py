import itertools

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from . import control, estimation, viz

fig, ax = plt.subplots()
(ln,) = ax.plot([], [])
plt.show(block=False)

env = gym.make("CartPole-v1", render_mode=None)

Q = np.zeros(shape=(100,) * 4 + (2,), dtype=np.float64)
M = np.zeros(
    shape=(100,) * 4 + (2,), dtype=np.uint32
)  # Monte Carlo incremental Average (+ 1/M * error)

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
    rewards = []
    states = []
    actions = []
    observation, info = env.reset()
    o_b = bin(observation)
    for step in itertools.count():
        states.append(o_b)
        action = control.action(env, o_b, Q)
        observation, reward, terminated, truncated, info = env.step(action)
        o_b = bin(observation)

        rewards.append(reward)
        actions.append(action)

        if truncated or terminated:
            break

    states = np.array(states)
    rewards = np.array(rewards)
    actions = np.array(actions)
    M, V = estimation.monte_carlo(rewards, states, actions, gamma, M, Q)

    print(f"ep {episode}")
    if episode % 1000 == 0:
        y = viz.value_by_angle(Q, M)
        x = np.arange(len(y))
        ln.set_data(x, y)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)


env.close()


def main():
    pass
