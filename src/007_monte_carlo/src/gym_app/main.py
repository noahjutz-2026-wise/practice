import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()
total_reward = 0

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if truncated or terminated:
        break

env.close()

print(total_reward)


def main():
    pass
