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
    n_runs = 2_000
    n_steps = 1_000
    configurations = [
        SimulationArgs(
            n_actions=10,
            epsilon=0.1,
            alpha=action_value_constant(0.1),
            n_steps=n_steps,
            initial_q=0,
            is_stationary=True,
        ),
        SimulationArgs(
            n_actions=10,
            epsilon=0,
            alpha=action_value_constant(0.1),
            n_steps=n_steps,
            initial_q=5,
            is_stationary=True,
        ),
    ]

    results = []
    for args in configurations:
        result = runs(args, n_runs)
        results.append(result)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    for i, dataset in enumerate(results):
        [data1, data2] = dataset
        ax1.plot(data1, label=i)
        ax2.plot(data2, label=i)
    ax1.legend()
    ax2.legend()
    ax1.set_title("Average Reward")
    ax2.set_title("% Optimal Action")
    plt.show()
