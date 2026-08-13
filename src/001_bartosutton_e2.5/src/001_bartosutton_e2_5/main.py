from typing import Callable

import matplotlib
import numpy as np
from numpy.typing import NDArray

Q_star = np.ones(10)  # rewards
Q = np.zeros(10)  # action value estimates

epsilon = 0.1
alpha = 0.9


def observe_reward(a):
    return np.random.normal(Q_star[a], 1)


def estimate_next_value(action, reward, alpha):
    return Q[action] + alpha * (reward - Q[action])


def run(
    epsilon: float, alpha: Callable[[int], float], n_steps: int
) -> NDArray[np.float64]:
    rewards = np.array(n_steps)
    for i in range(n_steps):
        alpha_n = alpha(i)
        is_explore = np.random.random() < epsilon
        if is_explore:
            a = np.random.randint(0, 10)
        else:
            a = np.argmax(Q)

        r = observe_reward(a)

        Q[a] = estimate_next_value(a, r, alpha_n)

        Q_star += np.random.normal(0, 0.1, 10)  # simulate nonstationary problem
        print(Q_star)

        # todo log values for plotting
    return rewards


run(epsilon, lambda n: alpha, 100)
