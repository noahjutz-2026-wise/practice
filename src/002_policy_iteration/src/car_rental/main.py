import itertools

import numpy as np
import scipy
from numpy.typing import NDArray

# requests = out
# returns = in
# Locations = (1, 2)

# given constants

gamma = 0.9
lambda_out = [3, 4]
lambda_in = [3, 2]

# 1. Initialization
pi = np.zeros((21, 21), dtype=np.int8)
v = np.zeros((21, 21))


def expected_return(s1: int, s2: int) -> float:
    a = pi[s1, s2]
    n1 = s1 - a
    n2 = s2 + a

    n_moved = abs(a)
    n_out_1 = scipy.stats.poisson.expect(
        lambda n: np.minimum(n1, n), args=(lambda_out[0],)
    )
    n_out_2 = scipy.stats.poisson.expect(
        lambda n: np.minimum(n2, n), args=(lambda_out[1],)
    )

    value = 0.0
    p_s1_ = [
        sum(
            scipy.stats.poisson.pmf(i, mu=lambda_in[0])
            * scipy.stats.poisson.pmf(j, mu=lambda_out[0])
            for i in range(12)
            for j in range(12)
            if s1_ == min(20, n1 - min(n1, j) + i)
        )
        for s1_ in range(21)
    ]
    p_s2_ = [
        sum(
            scipy.stats.poisson.pmf(i, mu=lambda_in[1])
            * scipy.stats.poisson.pmf(j, mu=lambda_out[1])
            for i in range(12)
            for j in range(12)
            if s2_ == min(20, n2 - min(n2, j) + i)
        )
        for s2_ in range(21)
    ]

    for s1_ in range(21):
        for s2_ in range(21):
            value += p_s1_[s1_] * p_s2_[s2_] * v[s1_, s2_]

    return -2 * n_moved + 10 * (n_out_1 + n_out_2) + gamma * value


# 2. Policy Evaluation
def eval(pi: NDArray[np.int8], v: NDArray[np.float64], theta: float = 0.1):
    for k in itertools.count(1):
        delta = 0
        for s1, s2 in np.ndindex(v.shape):
            s1 = int(s1)
            s2 = int(s2)
            v_old = v[s1, s2]
            v[s1, s2] = expected_return(s1, s2)
            delta = max(delta, abs(v_old - v[s1, s2]))
            print(f"(k={k}, delta={delta}) ({s1},{s2}):{v[s1, s2]}")

        if delta < theta:
            print(v)
            break


def main():
    eval(pi, v)
