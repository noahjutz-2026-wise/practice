import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

from . import control

env = gym.make("CartPole-v1", render_mode="human")
env = RecordEpisodeStatistics(env)

q = np.zeros(shape=(100, 100, 100, 100), dtype=np.uint8)

bins = np.vstack(
    (
        np.linspace(-4.8, 4.8, num=100),
        np.linspace(-5, 5, num=100),
        np.linspace(-0.418, 0.418, num=100),
        np.linspace(-5, 5, num=100),
    )
)


for episode in range(10):
    observation, info = env.reset()
    total_reward = 0
    while True:
        action = control.action(env, None)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if truncated or terminated:
            break

env.close()

print(f"\nEvaluation Summary:")
print(f"Episode durations: {list(env.time_queue)}")
print(f"Episode rewards: {list(env.return_queue)}")
print(f"Episode lengths: {list(env.length_queue)}")

# Calculate some useful metrics
avg_reward = np.average(env.return_queue)
avg_length = np.average(env.length_queue)
std_reward = np.std(env.return_queue)

print(f"\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}")
print(f"Average episode length: {avg_length:.1f} steps")
print(
    f"Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}"
)


def main():
    pass
