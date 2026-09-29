from collections import deque

import os
import pathlib
import time

import yaml
import torch.nn as nn
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

import wandb


class SuccessRateStopCallback(BaseCallback):
    def __init__(self, stats_channel: StatsSideChannel):
        super().__init__()
        self.stats_channel = stats_channel
        self.window = deque(maxlen=50)

    def _on_step(self) -> bool:
        for done, rew in zip(self.locals["dones"], self.locals["rewards"]):
            if done:
                is_goal = rew > 2.0
                self.window.append(1.0 if is_goal else 0.0)
                rate = sum(self.window) / len(self.window)

                stats = self.stats_channel.get_and_reset_stats()
                checkpoints = stats.get("achieved_checkpoints", [(0,)])[0][0]
                path_ratio = stats.get("path_completion_ratio", [(0,)])[0][0]

                print(
                    f"Step {self.num_timesteps:6d} | Goal: {str(is_goal):<5} | "
                    f"Path: {path_ratio * 100:4.1f}% (Checkpoints: {checkpoints:3.0f}) | "
                    f"Rew: {rew:5.2f} | Success Rate: {rate * 100:4.1f}% ({len(self.window)}/50)"
                )
                wandb.log(
                    {
                        "rollout/success_rate": rate,
                        "rollout/path_ratio": path_ratio,
                        "rollout/checkpoints": checkpoints,
                        "rollout/episode_reward": rew,
                    }
                )
                if len(self.window) == 50 and rate >= 0.75:
                    print("Reached 75% success rate. Stopping training.")
                    return False
        return True


def make_env(max_attempts=20, cooldown=2.0):
    # mlagents_envs picks base_port=5004 + 5*worker_id for worker 0, so random
    # worker_ids only give a random port; another process may still claim it
    # between check and bind, hence retry with a fresh worker_id/port.
    for attempt in range(max_attempts):
        worker_id = int(os.environ.get("SLURM_PROCID", os.getpid() % 100)) + attempt
        try:
            channel = EngineConfigurationChannel()
            channel.set_configuration_parameters(time_scale=20.0)
            stats_channel = StatsSideChannel()
            env = UnityToGymWrapper(
                UnityEnvironment(
                    "/home/noah/Downloads/export/unitybuild.x86_64",
                    worker_id=worker_id,
                    no_graphics=True,
                    side_channels=[channel, stats_channel],
                    additional_args=["-logFile", "mlagents.log"],
                )
            )
            return env, stats_channel
        except UnityWorkerInUseException:
            print(f"Port for worker {worker_id} in use, retrying ({attempt + 1}/{max_attempts})")
            time.sleep(cooldown)
    raise RuntimeError(f"Could not find a free Unity communication port after {max_attempts} attempts")


def load_params():
    params_path = pathlib.Path(__file__).resolve().parents[2] / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    config = {
        "user": "jno",
        "policy_type": "MlpPolicy",
        **params,
    }
    run = wandb.init(
        project="sb3",
        entity="tjno",
        config=config,
        sync_tensorboard=True,
        save_code=True,
    )

    env, stats_channel = make_env()

    policy_kwargs = dict(
        net_arch=dict(pi=[1024, 1024, 1024], vf=[1024, 1024, 1024]),
        activation_fn=nn.SiLU,
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        ent_coef=params["ent_coef"],
        clip_range=params["clip_range"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        verbose=0,
        tensorboard_log=f"runs/{run.id}",
    )
    callbacks = CallbackList(
        [
            SuccessRateStopCallback(stats_channel),
            WandbCallback(model_save_path=f"models/{run.id}", verbose=2),
        ]
    )
    model.learn(total_timesteps=params["total_timesteps"], callback=callbacks)
    model.save("unity_model")
    run.finish()
    env.close()


if __name__ == "__main__":
    main()
