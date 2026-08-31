import numpy as np
from gymnasium import Env
from numpy.typing import NDArray

State = tuple[int, int, int, int]


class Policy:
    """
    Epsilon-greedy policy w.r.t state-action values Q
    """

    def __init__(self, epsilon: float, env: Env, Q: NDArray):
        self.epsilon = epsilon
        self.random = env.np_random
        self.Q = Q

    def a(self, s: State) -> int:
        """
        Next action given state

        Args:
            s: State
        Returns:
            action in {0, 1}
        """
        if self.epsilon > 0 and self.random.random() < self.epsilon:
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
        p_e = self.epsilon / 2  # explore
        p_g = 1 - self.epsilon  # exploit
        is_greedy_a = a == np.argmax(self.Q[s], axis=-1)
        if is_greedy_a:
            return p_g + p_e
        else:
            return p_e

    def update(self, Q: NDArray) -> None:
        self.Q = Q

    def _greedy(self, s: State) -> int:
        q0, q1 = self.Q[tuple(s)]
        if q0 > q1:
            return 0
        elif q1 > q0:
            return 1
        return self.random.integers(2)
