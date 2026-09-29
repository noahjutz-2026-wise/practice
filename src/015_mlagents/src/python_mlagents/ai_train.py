from collections import deque

import inspect
import json
import os
import pathlib
import yaml

from mlagents_envs.environment import UnityEnvironment
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
        self.episode_path_ratios = []
        self.episode_rewards = []

    def _write_metrics(self):
        # Written continuously so even killed/failed runs leave comparable
        # numbers behind. DVC reads this file (declared in dvc.yaml).
        n = len(self.episode_path_ratios)
        metrics = {
            "mean_path_ratio": sum(self.episode_path_ratios) / n if n else 0.0,
            "mean_episode_reward": sum(self.episode_rewards) / n if n else 0.0,
            "success_rate_50": sum(self.window) / len(self.window) if self.window else 0.0,
            "episodes": n,
            "total_steps": self.num_timesteps,
        }
        metrics_path = pathlib.Path(__file__).resolve().parents[2] / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def _on_step(self) -> bool:
        for done, rew in zip(self.locals["dones"], self.locals["rewards"]):
            if done:
                is_goal = rew > 2.0
                self.window.append(1.0 if is_goal else 0.0)
                rate = sum(self.window) / len(self.window)

                stats = self.stats_channel.get_and_reset_stats()
                checkpoints = stats.get("achieved_checkpoints", [(0,)])[0][0]
                path_ratio = stats.get("path_completion_ratio", [(0,)])[0][0]
                self.episode_path_ratios.append(path_ratio)
                self.episode_rewards.append(rew)

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
                self._write_metrics()
                if len(self.window) == 50 and rate >= 0.75:
                    print("Reached 75% success rate. Stopping training.")
                    return False
        return True


def make_env():
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=20.0)
    stats_channel = StatsSideChannel()
    env = UnityToGymWrapper(
        UnityEnvironment(
            "/home/noah/Downloads/export/unitybuild.x86_64",
            # unique per process so parallel DVC experiments don't share a port
            worker_id=int(os.environ.get("SLURM_PROCID", os.getpid() % 100)),
            no_graphics=True,
            side_channels=[channel, stats_channel],
            additional_args=["-logFile", "mlagents.log"],
        )
    )
    return env, stats_channel


def load_params():
    params_path = pathlib.Path(__file__).resolve().parents[2] / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    # PPO defaults straight from the installed SB3 signature — nothing hardcoded
    ppo_defaults = {
        name: (None if p.default is inspect.Parameter.empty else p.default)
        for name, p in inspect.signature(PPO.__init__).parameters.items()
        if name not in ("self", "policy", "env", "kwargs")
    }
    config = {
        "user": "jno",
        "policy_type": "MlpPolicy",
        **params,
        "ppo_defaults": {k: str(v) for k, v in ppo_defaults.items()},
    }
    run = wandb.init(
        project="sb3",
        entity="tjno",
        config=config,
        sync_tensorboard=True,
        save_code=True,
    )

    env, stats_channel = make_env()

    # only what params.yaml overrides; everything else = SB3 defaults
    ppo_kwargs = {k: v for k, v in params.items() if k != "total_timesteps"}
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=f"runs/{run.id}", **ppo_kwargs)
    run.config.update(
        {
            # actual resolved settings read from the live policy, not hardcoded
            "activation_fn": model.policy.activation_fn.__name__,
            "net_arch": str(model.policy.net_arch),
        }
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
