import jax.numpy as jnp
import matplotlib.pyplot as plt


def main():
    x_jnp = jnp.linspace(0, 10, 1000)
    y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
    print(x_jnp.devices())
    plt.plot(x_jnp, y_jnp)
    plt.show()
