from collections import deque

import torch.nn as nn
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

BUILD_PATH = "/home/noah/Downloads/export/unitybuild.x86_64"
NUM_ENVS = 4


class StatsUnityWrapper(UnityToGymWrapper):
    def __init__(self, unity_env: UnityEnvironment, stats_channel: StatsSideChannel):
        super().__init__(unity_env)
        self.stats_channel = stats_channel

    def step(self, action):
        obs, rew, done, info = super().step(action)
        if done:
            stats = self.stats_channel.get_and_reset_stats()
            info["checkpoints"] = stats.get("achieved_checkpoints", [(0,)])[0][0]
            info["path_ratio"] = stats.get("path_completion_ratio", [(0,)])[0][0]
        return obs, rew, done, info


class SuccessRateStopCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.window = deque(maxlen=50)

    def _on_step(self) -> bool:
        for done, rew, info in zip(
            self.locals["dones"], self.locals["rewards"], self.locals["infos"]
        ):
            if done:
                is_goal = rew > 2.0
                self.window.append(1.0 if is_goal else 0.0)
                rate = sum(self.window) / len(self.window)
                checkpoints = info.get("checkpoints", 0)
                path_ratio = info.get("path_ratio", 0.0)

                print(
                    f"Step {self.num_timesteps:6d} | Goal: {str(is_goal):<5} | "
                    f"Path: {path_ratio * 100:4.1f}% (Checkpoints: {checkpoints:3.0f}) | "
                    f"Rew: {rew:5.2f} | Success Rate: {rate * 100:4.1f}% ({len(self.window)}/50)"
                )
                if len(self.window) == 50 and rate >= 0.75:
                    print("Reached 75% success rate. Stopping training.")
                    return False
        return True


def make_env(worker_id: int):
    def _init():
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=20.0)
        stats_channel = StatsSideChannel()
        unity_env = UnityEnvironment(
            BUILD_PATH,
            worker_id=worker_id,
            no_graphics=True,
            side_channels=[channel, stats_channel],
            additional_args=["-logFile", "mlagents.log"],
        )
        return StatsUnityWrapper(unity_env, stats_channel)

    return _init


def main():
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

    policy_kwargs = dict(
        net_arch=dict(pi=[1024, 1024, 1024], vf=[1024, 1024, 1024]),
        activation_fn=nn.SiLU,
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        ent_coef=0.01,
        verbose=0,
    )
    model.learn(total_timesteps=1_000_000, callback=SuccessRateStopCallback())
    model.save("unity_model")
    env.close()


if __name__ == "__main__":
    main()

