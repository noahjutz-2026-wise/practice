import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from rlsim.simulation import Simulation


def sample_average(n: int):
    n += 1  # 0-indexing to 1-indexing
    return 1 / n


def runs(simulation: Simulation, n_runs: int) -> NDArray[np.float64]:
    average = np.zeros(simulation.n_steps)
    for i in range(n_runs):
        average += simulation.run()
        print(i)
    average /= n_runs
    return average


def main():
    sim = Simulation(n_actions=10, epsilon=0.1, alpha=sample_average, n_steps=10_000)
    data = runs(sim, 1_000)
    plt.plot(data)
    plt.show()
