import gymnasium as gym
import numpy as np

from . import control


def discretize(observation):
    o_binned = (observation[:, None] >= bins).sum(axis=1)
    o_discrete = bins[np.arange(4), o_binned]
    return (o_binned, o_discrete)


env = gym.make("CartPole-v1", render_mode="human")

q = np.zeros(shape=(100, 100, 100, 100), dtype=np.uint8)

bins = np.vstack(
    (
        np.linspace(-4.8, 4.8, num=100),
        np.linspace(-5, 5, num=100),
        np.linspace(-0.418, 0.418, num=100),
        np.linspace(-5, 5, num=100),
    )
)

observation, info = env.reset()
o_b, o_d = discretize(observation)
total_reward = 0

while True:
    action = control.action(env, o_d)
    observation, reward, terminated, truncated, info = env.step(action)
    # state = observation
    total_reward += reward
    if truncated or terminated:
        break

env.close()

print(total_reward)


def main():
    pass
