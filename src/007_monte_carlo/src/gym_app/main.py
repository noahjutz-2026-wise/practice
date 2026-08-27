import itertools

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pyinstrument
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

bin_lo = np.array([-4.8, -5.0, -0.418, -5.0])
bin_hi = np.array([4.8, 5.0, 0.418, 5.0])
n_bins_arr = np.array(n_bins, dtype=np.float64)
bin_step = (bin_hi - bin_lo) / (n_bins_arr - 1)
bin_max = np.array(n_bins, dtype=np.int64)  # max index = n_bins (one past last edge)


def bin(observation: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.clip(
        ((observation - bin_lo) / bin_step + 0.5).astype(np.int64), 0, bin_max
    )


def resolve(b: NDArray[np.int64]) -> NDArray[np.float64]:
    return bin_lo + b * bin_step


p = pyinstrument.Profiler()
p.start()
for episode in range(10000):
    if episode == 50000:
        env = gym.make("CartPole-v1", render_mode="human")
    rewards = []
    states = []
    actions = []
    observation, info = env.reset()
    o_b = bin(observation)
    for step in itertools.count():
        states.append(o_b)
        action = control.action(o_b, Q, epsilon)
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

p.stop()
p.print()
p.open_in_browser()

env.close()


def main():
    pass
