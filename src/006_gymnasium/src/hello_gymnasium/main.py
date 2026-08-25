import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")


for i in range(200):
    total_reward = 0
    observation, info = env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if truncated or terminated:
            break
    print(total_reward)

env.close()


def main():
    pass
