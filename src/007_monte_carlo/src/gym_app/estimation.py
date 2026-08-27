import numpy as np
from numpy.typing import NDArray


def isr(
    epsilon: float, Q: NDArray, actions: NDArray, states: NDArray, t: int, T_1: int
) -> float:
    """
    Importance-Sampling Ratio (Ordinary Importance Sampling) for
    - an epsilon-greedy behavior policy b
    - a greedy target policy pi
    with the same state-action values Q. Given an episode with actions, states.
    """

    A = 2  # amount of actions A(s)

    # Target Policy (greedy)
    def pi(a: int, s: tuple[int, int, int, int]):
        return a == np.argmax(Q[s], axis=-1)

    # Behavior-policy (epsilon-greedy)
    def b(a: int, s: tuple[int, int, int, int]):
        p_e = epsilon / A  # explore
        p_g = 1 - epsilon  # exploit
        p_pi = pi(a, s)
        if p_pi == 1:  # a=a*
            return p_g + p_e
        elif p_pi == 0:  # a!=a*
            return p_e

    def ratio(i):
        a = actions[i]
        s = tuple(states[i])
        return pi(a, s) / b(a, s)

    rho = np.fromfunction(ratio, (T_1 - t,))
    return np.prod(rho)


def monte_carlo(
    rewards: NDArray,
    states: NDArray,
    actions: NDArray,
    gamma: float,
    M: NDArray,
    Q: NDArray,
) -> tuple[NDArray, NDArray]:
    """
    First-Visit MC prediction, given one episode
    """
    T = len(rewards)
    # Build set of first-visit timesteps: for each (state, action) pair,
    # only the earliest timestep counts.
    first_visit: dict[tuple, int] = {}
    for t in range(T):
        key = (*states[t], actions[t])
        if key not in first_visit:
            first_visit[key] = t

    # Backward pass to compute returns
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = gamma * G + rewards[t]
        key = (*states[t], actions[t])
        if first_visit[key] == t:
            idx = tuple(key)
            M[idx] += 1
            Q[idx] += (G - Q[idx]) / M[idx]
    return M, Q
