import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def value_by_angle(Q: NDArray, M: NDArray) -> NDArray:
    # return np.mean(V, axis=(0, 1, 3))
    return np.where(M > 0, Q, 0.0).sum(axis=(0, 1, 3, 4)) / np.maximum(
        (M > 0).sum(axis=(0, 1, 3, 4)), 1
    )
