import itertools

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import wandb
from gym_app.env import DiscreteCartPole

from . import control, estimation


def train(run: wandb.Run, Q: NDArray | None = None) -> NDArray:
    """
    General Policy Iteration Loop

    Args:
        Q: (*n_bins, 2) initial state-action values
    Returns:
        Q: (*n_bins, 2) state-action values after training
    """
    n_bins = tuple(run.config["n_bins"])
    gamma = run.config["gamma"]
    epsilon = run.config["epsilon"]
    episodes = run.config["episodes"]
    log_every = run.config["log_every"]
    task = run.config["task"]
    prediction_method = run.config["prediction_method"]

    env = gym.make("CartPole-v1", render_mode=None)
    env = DiscreteCartPole(env, n_bins)

    if Q is None:
        Q = np.zeros(shape=n_bins + (2,), dtype=np.float64)
    C = np.zeros(
        shape=n_bins + (2,), dtype=np.float64
    )  # Monte Carlo incremental Average (+ 1/M * error)

    for episode in range(episodes):
        rewards = []
        states = []
        actions = []
        observation, info = env.reset()
        for step in itertools.count():
            states.append(observation)
            action = control.b(observation, Q, epsilon)
            observation, reward, terminated, truncated, info = env.step(action)

            rewards.append(reward)
            actions.append(action)

            if truncated or terminated:
                break

        states = np.array(states)
        rewards = np.array(rewards)
        actions = np.array(actions)
        if episode % log_every == 0:
            last_Q = Q.copy()
        if task == "train":
            C, Q = estimation.monte_carlo(
                rewards, states, actions, gamma, epsilon, C, Q
            )

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
