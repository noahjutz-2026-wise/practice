import itertools

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import wandb

from . import control, estimation


def train(run: wandb.Run) -> NDArray:
    n_bins = tuple(run.config["n_bins"])
    gamma = run.config["gamma"]
    epsilon = run.config["epsilon"]
    episodes = run.config["episodes"]
    log_every = run.config["log_every"]

    env = gym.make("CartPole-v1", render_mode=None)

    Q = np.zeros(shape=n_bins + (2,), dtype=np.float64)
    C = np.zeros(
        shape=n_bins + (2,), dtype=np.float64
    )  # Monte Carlo incremental Average (+ 1/M * error)

    bin_lo = np.array([-4.8, -5.0, -0.418, -5.0])
    bin_hi = np.array([4.8, 5.0, 0.418, 5.0])
    n_bins_arr = np.array(n_bins, dtype=np.float64)
    bin_step = (bin_hi - bin_lo) / (n_bins_arr - 1)
    bin_max = np.array(
        n_bins, dtype=np.int64
    )  # max index = n_bins (one past last edge)

    def bin(observation: NDArray[np.float64]) -> NDArray[np.int64]:
        return np.clip(
            ((observation - bin_lo) / bin_step + 0.5).astype(np.int64), 0, bin_max
        )

    def resolve(b: NDArray[np.int64]) -> NDArray[np.float64]:
        return bin_lo + b * bin_step

    for episode in range(episodes):
        # if episode == episodes - 10:
        #     env = gym.make("CartPole-v1", render_mode="human")
        rewards = []
        states = []
        actions = []
        observation, info = env.reset()
        o_b = bin(observation)
        for step in itertools.count():
            states.append(o_b)
            action = control.b(o_b, Q, epsilon)
            observation, reward, terminated, truncated, info = env.step(action)
            o_b = bin(observation)

            rewards.append(reward)
            actions.append(action)

            if truncated or terminated:
                break

        states = np.array(states)
        rewards = np.array(rewards)
        actions = np.array(actions)
        if episode % log_every == 0:
            last_Q = Q.copy()
        C, Q = estimation.monte_carlo(rewards, states, actions, gamma, epsilon, C, Q)

        if episode % log_every == 0:
            run.log(
                {
                    "episode": episode,
                    "cum_reward": rewards.sum(),
                    "steps": step + 1,
                    "q_coverage": (C > 0).sum(),
                    "q_value": Q[C > 0].mean(),
                    "stability": np.count_nonzero(Q != last_Q),
                }
            )

    env.close()

    return Q
