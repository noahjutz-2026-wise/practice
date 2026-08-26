import gymnasium as gym
import numpy as np

from . import control

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
total_reward = 0

while True:
    action = control.action(env, None)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if truncated or terminated:
        break

env.close()

print(total_reward)


def main():
    pass
