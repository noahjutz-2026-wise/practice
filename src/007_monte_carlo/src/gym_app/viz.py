import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def value_by_angle(q: NDArray[np.uint8]) -> NDArray:
    return np.mean(q, axis=(0, 1, 3))
