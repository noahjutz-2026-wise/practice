import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def value_by_angle(Q: NDArray, M: NDArray) -> NDArray:
    # Returns shape (100, 2) — one column per action
    return np.where(M > 0, Q, 0.0).sum(axis=(0, 1, 3)) / np.maximum(
        (M > 0).sum(axis=(0, 1, 3)), 1
    )
