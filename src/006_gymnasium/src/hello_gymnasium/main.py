import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += float(reward)
    episode_over = truncated or terminated

print(total_reward)
env.close()


def main():
    pass
