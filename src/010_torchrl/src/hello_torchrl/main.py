from torchrl.envs import GymEnv, step_mdp


def main():
    env = GymEnv("Pendulum-v1")
    reset = env.reset()
    reset_with_action = env.rand_action(reset)
    stepped_data = env.step(reset_with_action)
    data = step_mdp(stepped_data)
    print(data)
