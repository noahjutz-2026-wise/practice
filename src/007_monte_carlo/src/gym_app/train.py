import itertools

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

import wandb
from gym_app.env import DiscreteCartPole

from . import control, estimation


def train(run: wandb.Run, Q: NDArray | None = None) -> NDArray:
    """
    General Policy Iteration Loop.

    Observation and state are treated as equal.

    Args:
        Q: (*n_bins, 2) initial state-action values
    Returns:
        Q: (*n_bins, 2) state-action values after training
    """
    n_bins = tuple(run.config["n_bins"])
    gamma = run.config["gamma"]
    epsilon = run.config["epsilon"]
    alpha = run.config["alpha"]
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
        cum_reward = 0
        rewards = []
        states = []
        actions = []
        observation, info = env.reset()
        action = control.b(observation, Q, 1)
        for t in itertools.count():
            new_observation, reward, terminated, truncated, info = env.step(action)
            new_action = control.b(new_observation, Q, 1 / (t + 1))

            match prediction_method:
                case "monte_carlo":
                    states.append(observation)
                    rewards.append(reward)
                    actions.append(action)
                case "sarsa":
                    Q = estimation.sarsa(
                        Q,
                        alpha,
                        gamma,
                        (*observation, action),
                        (*new_observation, new_action),
                        reward,
                        truncated or terminated,
                    )

            observation = new_observation
            action = new_action

            cum_reward += reward

            if truncated or terminated:
                break

        states = np.array(states)
        rewards = np.array(rewards)
        actions = np.array(actions)

        if episode % log_every == 0:
            last_Q = Q.copy()

        if task == "train" and prediction_method == "monte_carlo":
            C, Q = estimation.monte_carlo(
                rewards, states, actions, gamma, epsilon, C, Q
            )

        if episode % log_every == 0:
            run.log(
                {
                    "episode": episode,
                    "cum_reward": cum_reward,
                    "steps": t + 1,
                    "q_coverage": (C > 0).sum(),
                    "q_value": Q[C > 0].mean(),
                    "stability": np.count_nonzero(Q != last_Q),
                }
            )

    env.close()

    return Q
