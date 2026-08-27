import itertools

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from . import control, estimation, viz

config = {"n_bins": (15, 15, 15, 15), "gamma": 0.9, "epsilon": 0.01}
n_bins = config["n_bins"]
gamma = config["gamma"]
epsilon = config["epsilon"]

fig, ax = plt.subplots()
(ln0,) = ax.plot([], [], label="action 0 (left)")
(ln1,) = ax.plot([], [], label="action 1 (right)")
ax.legend()
plt.show(block=False)

env = gym.make("CartPole-v1", render_mode=None)


Q = np.zeros(shape=n_bins + (2,), dtype=np.float64)
M = np.zeros(
    shape=n_bins + (2,), dtype=np.uint32
)  # Monte Carlo incremental Average (+ 1/M * error)

bins = np.vstack(
    (
        np.linspace(-4.8, 4.8, num=n_bins[0]),
        np.linspace(-5, 5, num=n_bins[1]),
        np.linspace(-0.418, 0.418, num=n_bins[2]),
        np.linspace(-5, 5, num=n_bins[3]),
    )
)


def bin(observation: NDArray[np.float64]) -> NDArray[np.uint8]:
    return (observation[:, None] >= bins).sum(axis=1)


def resolve(bin: NDArray[np.uint8]) -> NDArray[np.float64]:
    return bins[np.arange(4), bin]  # todo out of bounds exception


for episode in itertools.count():
    if episode == 50000:
        env = gym.make("CartPole-v1", render_mode="human")
    rewards = []
    states = []
    actions = []
    observation, info = env.reset()
    o_b = bin(observation)
    for step in itertools.count():
        states.append(o_b)
        action = control.action(env, o_b, Q, epsilon)
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

    if episode % 1000 == 0:
        print(f"ep {episode}")
    if episode % 20000 == 0:
        y = viz.value_by_angle(Q, M)
        x = np.arange(y.shape[0])
        ln0.set_data(x, y[:, 0])
        ln1.set_data(x, y[:, 1])
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)


env.close()


def main():
    pass
