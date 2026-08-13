import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from rlsim.simulation import Simulation
from rlsim.simulation_args import SimulationArgs


def action_value_sample_average(n: int):
    n += 1  # 0-indexing to 1-indexing
    return 1 / n


def action_value_constant(alpha: float):
    def alpha_n(_n: int):
        return alpha

    return alpha_n


def runs(args: SimulationArgs, n_runs: int) -> NDArray[np.float64]:
    average = np.zeros(args.n_steps)
    for i in range(n_runs):
        sim = Simulation(args)
        average += sim.run()
        print(i)
    average /= n_runs
    return average


def main():
    args = SimulationArgs(
        n_actions=10, epsilon=0.1, alpha=action_value_constant(0.1), n_steps=10_000
    )
    data = runs(args, 1_000)
    plt.plot(data)
    plt.show()
