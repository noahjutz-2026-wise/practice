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


# 2. Policy Evaluation
def eval(pi: NDArray[np.int8], v: NDArray[np.float64], theta=0.1):
    for s1, s2 in np.ndindex(v.shape):
        v_old = v[s1, s2]
        a = pi[s1, s2]

        cost = -2 * abs(a)
        n_lent_1 = scipy.stats.poisson.expect(
            lambda n: np.minimum(s1, n), args=(lambda_out[0],)
        )
        n_lent_2 = scipy.stats.poisson.expect(
            lambda n: np.minimum(s2, n), args=(lambda_out[1])
        )
        print(revenue_1)
        exit(0)


def main():
    eval(pi, v)
