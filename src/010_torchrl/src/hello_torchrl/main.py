import torch
from ray.rllib.algorithms.ppo import PPOConfig
from tensordict.nn import TensorDictModule
from torchrl.envs import GymEnv, StepCounter, TransformedEnv, step_mdp


def main():
    env = GymEnv("Pendulum-v1")
    env = TransformedEnv(env, StepCounter(max_steps=10))

    module = torch.nn.LazyLinear(out_features=env.action_spec.shape[-1])
    policy = TensorDictModule(module, in_keys=["observation"], out_keys=["action"])

    rollout = env.rollout(max_steps=100, policy=policy)
    print(rollout)
