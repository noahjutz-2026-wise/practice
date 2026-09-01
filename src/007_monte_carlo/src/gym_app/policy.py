import numpy as np
from gymnasium import Env
from numpy.typing import NDArray

State = tuple[int, int, int, int]


class Policy:
    """
    Epsilon-greedy policy w.r.t state-action values Q
    """

    def __init__(self, epsilon: float, env: Env, Q: NDArray):
        """
        Args:
            epsilon: Exploration rate. 1=random, 0=deterministic, None=1/t
            env: Gymnasium environment
            Q: state-action values
        """
        self.epsilon = epsilon
        self.random = env.np_random
        self.Q = Q
        self._episode = 1

    def a(self, s: State) -> int:
        """
        Next action given state

        Args:
            s: State
        Returns:
            action in {0, 1}
        """
        if (
            self._exploration_rate() > 0
            and self.random.random() < self._exploration_rate()
        ):
            return self.random.integers(2)
        return self._greedy(s)

    def p(self, a: int, s: State) -> float:
        """
        Probability distribution of policy

        Args:
            a: Action
            s: State
        Returns:
            Pr(a | s) in [0, 1]
        """
        p_e = self._exploration_rate() / 2  # explore
        p_g = 1 - self._exploration_rate()  # exploit
        is_greedy_a = a == np.argmax(self.Q[tuple(s)], axis=-1)
        if is_greedy_a:
            return p_g + p_e
        else:
            return p_e

    def update(self, Q: NDArray) -> None:
        self.Q = Q

    def step_episode(self) -> None:
        self._episode += 1

    def reset(self, env: Env) -> None:
        self.random = env.np_random

    def _greedy(self, s: State) -> int:
        q0, q1 = self.Q[tuple(s)]
        if q0 > q1:
            return 0
        elif q1 > q0:
            return 1
        return self.random.integers(2)

    def _exploration_rate(self) -> float:
        if self.epsilon is not None:
            return float(self.epsilon)
        return 1 / self._episode
