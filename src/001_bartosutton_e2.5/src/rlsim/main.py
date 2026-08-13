from rlsim.simulation import Simulation


def main():
    simulation = Simulation(10)
    epsilon = 0.1
    alpha = 0.1
    n_steps = 10_000
    result = simulation.run(epsilon, lambda n: alpha, n_steps)
    print(result)
