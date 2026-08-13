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
    average = np.zeros((2, args.n_steps))
    for i in range(n_runs):
        sim = Simulation(args)
        average += sim.run()
        print(i)
    average /= n_runs
    return average


def main():
    n_runs = 1_000
    args_constant = SimulationArgs(
        n_actions=10, epsilon=0.1, alpha=action_value_constant(0.1), n_steps=10_000
    )
    args_sample_average = SimulationArgs(
        n_actions=10, epsilon=0.1, alpha=action_value_sample_average, n_steps=10_000
    )
    data_constant = runs(args_constant, n_runs)
    data_sample_average = runs(args_sample_average, n_runs)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    for dataset in [data_constant, data_sample_average]:
        [data1, data2] = dataset
        ax1.plot(data1, label="data1")
        ax2.plot(data2, label="data2")
    ax1.legend()
    ax2.legend()
    plt.show()
