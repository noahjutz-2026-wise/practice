from simulation import Simulation

simulation = Simulation(10)
epsilon = 0.1
alpha = 0.9
n_steps = 100
result = simulation.run(epsilon, lambda n: alpha, n_steps)
print(result)
