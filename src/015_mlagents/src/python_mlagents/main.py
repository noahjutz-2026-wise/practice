from collections import deque
import torch.nn as nn
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class SuccessRateStopCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.window = deque(maxlen=50)

    def _on_step(self) -> bool:
        for done, rew in zip(self.locals["dones"], self.locals["rewards"]):
            if done:
                is_goal = rew > 2.0
                self.window.append(1.0 if is_goal else 0.0)
                rate = sum(self.window) / len(self.window)
                print(f"Step {self.num_timesteps}: Goal={is_goal}, Rolling Rate={rate * 100:.1f}% ({len(self.window)}/50)")
                if len(self.window) == 50 and rate >= 0.75:
                    print("Reached 75% success rate. Stopping training.")
                    return False
        return True


def main():
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=20.0)
    env = UnityToGymWrapper(
        UnityEnvironment("/home/noah/Downloads/export/unitybuild.x86_64", no_graphics=True, side_channels=[channel])
    )

    policy_kwargs = dict(net_arch=dict(pi=[1024, 1024, 1024], vf=[1024, 1024, 1024]), activation_fn=nn.SiLU)
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=3e-4, n_steps=2048, batch_size=128, verbose=1)
    model.learn(total_timesteps=1_000_000, callback=SuccessRateStopCallback())
    model.save("unity_model")
    env.close()


if __name__ == "__main__":
    main()
