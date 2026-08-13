import matplotlib
import numpy as np

Q_star = np.ones(10)  # rewards
Q = np.zeros(10)  # action value estimates

epsilon = 0.1
alpha = 0.9


def observe_reward(a):
    return np.random.normal(Q_star[a], 1)


def estimate_next_value(action, reward, alpha):
    return Q[action] + alpha * (reward - Q[action])


for i in range(10_000):
    alpha_n = alpha  # constant
    is_explore = np.random.random() < epsilon
    if is_explore:
        a = np.random.randint(0, 10)
    else:
        a = np.argmax(Q)

    r = observe_reward(a)

    Q[a] = estimate_next_value(a, r, alpha_n)

    # todo walk q (rewards)
    # todo log values for plotting
