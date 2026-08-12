import matplotlib
import numpy as np

action_values = np.zeros(10)

q = np.ones(10)

epsilon = 0.1

alpha = 0.9


def reward(a):
    return np.random.normal(q[a], 1)


for i in range(10_000):
    is_explore = np.random.random() < epsilon
    if is_explore:
        a = np.random.randint(0, 10)
    else:
        a = np.argmax(action_values)

    r = reward(a)

    # todo compute action value
    # todo walk q (rewards)
    # todo log values for plotting
