import numpy as np
from numpy.typing import NDArray


# Epsilon-greedy policy w.r.t state-action values (specialized for 2 actions)
def action(state, Q: NDArray, epsilon: float):
    if np.random.random() < epsilon:
        return np.random.randint(2)
    q0, q1 = Q[tuple(state)]
    if q0 > q1:
        return 0
    elif q1 > q0:
        return 1
    return np.random.randint(2)
