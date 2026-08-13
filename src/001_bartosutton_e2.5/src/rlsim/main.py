import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from rlsim.simulation import Simulation
from rlsim.simulation_args import SimulationArgs


def action_value_sample_average(n: int):
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
    args_constant = SimulationArgs(
        n_actions=10, epsilon=0.1, alpha=action_value_constant(0.1), n_steps=10_000
    )
    args_sample_average = SimulationArgs(
        n_actions=10, epsilon=0.1, alpha=action_value_sample_average, n_steps=10_000
    )
    data_constant = runs(args_constant, 1_000)
    data_sample_average = runs(args_sample_average, 1_000)
    plt.plot(data_constant, label="Constant")
    plt.plot(data_sample_average, label="Sample Average")
    plt.show()
