import gymnasium as gym


# Random policy
def action(env: gym.Env, state):
    return env.action_space.sample()
