import jax.numpy as jnp
import numpy as np
from jax import jit


def norm(X):
    X = X - X.mean(0)
    return X / X.std(0)


def main():
    norm_compiled = jit(norm)
    np.random.seed(1701)
    X = jnp.array(np.random.rand(10000, 10))
    r = np.allclose(norm(X), norm_compiled(X), atol=1e-6)
    print(r)
