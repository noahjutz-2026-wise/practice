from torchrl.envs import GymEnv, StepCounter, TransformedEnv, step_mdp


def main():
    env = GymEnv("Pendulum-v1")
    env = TransformedEnv(env, StepCounter(max_steps=10))
    data = env.rollout(max_steps=100)
    print(data["next"]["truncated"])
