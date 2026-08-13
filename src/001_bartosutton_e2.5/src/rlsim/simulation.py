from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class Simulation:
    def __init__(
        self,
        n_actions: int,
        epsilon: float,
        alpha: Callable[[int], float],
        n_steps: int,
    ) -> None:
        self.Q_star = np.ones(n_actions)  # rewards
        self.Q = np.zeros(n_actions)  # action value estimates
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_steps = n_steps

    def observe_reward(self, a: int):
        return np.random.normal(self.Q_star[a], 1)

    def estimate_next_value(self, action, reward, alpha):
        return self.Q[action] + alpha * (reward - self.Q[action])

    def run(self) -> NDArray[np.float64]:
        rewards = np.zeros(self.n_steps)
        for i in range(self.n_steps):
            alpha_n = self.alpha(i)
            is_explore = np.random.random() < self.epsilon
            if is_explore:
                a = np.random.randint(0, 10)
            else:
                a = int(np.argmax(self.Q))

            r = self.observe_reward(a)

            self.Q[a] = self.estimate_next_value(a, r, alpha_n)

            # simulate nonstationary problem
            self.Q_star += np.random.normal(0, 0.1, 10)

            rewards[i] = r
        return rewards
