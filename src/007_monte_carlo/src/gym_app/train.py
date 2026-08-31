import itertools

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

import wandb
from gym_app.env import DiscreteCartPole
from gym_app.policy import Policy

from . import prediction


def train(run: wandb.Run, Q1: NDArray | None = None) -> NDArray:
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
    seed = run.config["seed"]

    env = gym.make("CartPole-v1", render_mode=None)
    env = DiscreteCartPole(env, n_bins)

    Q2 = np.zeros(shape=n_bins + (2,), dtype=np.float64)

    if Q1 is None:
        Q1 = np.zeros(shape=n_bins + (2,), dtype=np.float64)
    C = np.zeros(
        shape=n_bins + (2,), dtype=np.float64
    )  # Monte Carlo incremental Average (+ 1/M * error)

    pi = Policy(0, env, Q1 + Q2)
    b = Policy(epsilon, env, Q1 + Q2)

    last_Q1 = Q1.copy()

    for episode in range(episodes):
        pi.step_episode()
        b.step_episode()
        cum_reward = 0
        t = 1
        rewards = []
        states = []
        actions = []
        observation, info = env.reset(seed=seed)
        action = b.a(tuple(observation))
        for step in itertools.count():
            pi.update(Q1 + Q2)
            b.update(Q1 + Q2)
            new_observation, reward, terminated, truncated, info = env.step(action)
            new_action = b.a(tuple(new_observation))

            match prediction_method:
                case "monte_carlo":
                    states.append(observation)
                    rewards.append(reward)
                    actions.append(action)
                case "sarsa":
                    Q1 = prediction.sarsa(
                        Q1,
                        alpha,
                        gamma,
                        (*observation, action),
                        (*new_observation, new_action),
                        reward,
                        truncated or terminated,
                    )
                case "q_learning":
                    Q1, _ = prediction.q_learning(
                        Q1,
                        alpha,
                        gamma,
                        (*observation, action),
                        (*new_observation, new_action),
                        reward,
                        truncated or terminated,
                        env.np_random,
                    )
                case "expected_sarsa":
                    Q1 = prediction.expected_sarsa(
                        Q1,
                        alpha,
                        gamma,
                        (*observation, action),
                        (*new_observation, new_action),
                        reward,
                        truncated or terminated,
                        pi,
                    )
                case "double_q_learning":
                    Q1, Q2 = prediction.q_learning(
                        Q1,
                        alpha,
                        gamma,
                        (*observation, action),
                        (*new_observation, new_action),
                        reward,
                        truncated or terminated,
                        env.np_random,
                        Q2=Q2,
                    )

            observation = new_observation
            action = new_action

            cum_reward += reward
            t += 1

            if truncated or terminated:
                break

        states = np.array(states)
        rewards = np.array(rewards)
        actions = np.array(actions)

        if task == "train" and prediction_method == "monte_carlo":
            C, Q1 = prediction.monte_carlo(
                rewards, states, actions, gamma, epsilon, C, Q1, pi, b
            )

        if episode % log_every == 0:
            visited = (Q1 + Q2) != 0
            run.log(
                {
                    "episode": episode,
                    "cum_reward": cum_reward,
                    "steps": step + 1,
                    "q_coverage": visited.sum() / Q1.size,
                    "q_value": (Q1 + Q2)[visited].mean() if visited.any() else 0.0,
                    "stability": np.count_nonzero(Q1 != last_Q1),
                    "exploration_rate": b._exploration_rate(),
                }
            )
            last_Q1 = Q1.copy()

    env.close()

    return Q1 + Q2
