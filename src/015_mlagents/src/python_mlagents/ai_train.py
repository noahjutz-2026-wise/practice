from collections import deque

import inspect
import json
import os
import pathlib
import re
import socket
import yaml

import numpy as np

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

METRICS_KEYS = (
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/explained_variance",
)


class TrainMetricsCallback(BaseCallback):
    """Per-time-step metrics: losses, instantaneous reward, observation stats,
    plus the latest episode-granular success/path values. wandb.log is called
    every `log_interval_steps` steps (1:1 logging for 100k steps is wasteful —
    the values themselves only change on episode/update boundaries)."""

    def __init__(self, stats_channel: StatsSideChannel, log_interval_steps: int):
        super().__init__()
        self.stats_channel = stats_channel
        self.log_interval_steps = max(1, int(log_interval_steps))
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

    def _log_timestep(self):
        m = {"timestep": self.num_timesteps}
        logger = self.model.logger
        for key in METRICS_KEYS:
            if key in logger.name_to_value:
                m[key.split("/")[1]] = float(logger.name_to_value[key])
        m["step_reward"] = float(np.mean(self.locals["rewards"]))
        m["obs_mean"] = float(np.mean(self.locals["new_obs"]))
        m["obs_std"] = float(np.std(self.locals["new_obs"]))
        if self.window:
            m["success_rate_50"] = sum(self.window) / len(self.window)
        if self.episode_path_ratios:
            m["last_path_ratio"] = self.episode_path_ratios[-1]
        wandb.log(m, step=self.num_timesteps)

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
                        "episode/success_rate": rate,
                        "episode/path_ratio": path_ratio,
                        "episode/checkpoints": checkpoints,
                        "episode/reward": rew,
                    },
                    step=self.num_timesteps,
                )
                self._write_metrics()
                if len(self.window) == 50 and rate >= 0.75:
                    print("Reached 75% success rate. Stopping training.")
                    return False
        if self.num_timesteps % self.log_interval_steps == 0:
            self._log_timestep()
        return True


def make_env(time_scale=20.0):
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=time_scale)
    stats_channel = StatsSideChannel()
    # claim a free port up front so parallel DVC experiments never collide
    # (worker_id 0 => port = base_port; pid-based ids collide mod 100)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        base_port = s.getsockname()[1]
    env = UnityToGymWrapper(
        UnityEnvironment(
            "/home/noah/Downloads/export/unitybuild.x86_64",
            base_port=base_port,
            worker_id=0,
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


def wandb_group():
    # Batch label for wandb's group panel: explicit env override wins, else
    # derive from the DVC queue name (DVC_EXP_NAME="hpsearch-14" -> "hpsearch"),
    # so every experiment queued in one `dvc exp run --queue -n <name>` batch
    # lands in the same wandb group.
    group = os.environ.get("WANDB_GROUP")
    if group:
        return group
    exp_name = os.environ.get("DVC_EXP_NAME")
    if exp_name:
        return re.sub(r"-\d+$", "", exp_name)
    return None


def upload_model(run, model_path):
    model_file = pathlib.Path(model_path)
    if model_file.suffix != ".zip":
        model_file = pathlib.Path(model_path + ".zip")
    artifact = wandb.Artifact(f"{run.id}-model", type="model")
    artifact.add_file(str(model_file))
    run.log_artifact(artifact)


def run_train(params):
    # PPO defaults straight from the installed SB3 signature — nothing hardcoded
    ppo_defaults = {
        name: (None if p.default is inspect.Parameter.empty else p.default)
        for name, p in inspect.signature(PPO.__init__).parameters.items()
        if name not in ("self", "policy", "env", "kwargs")
    }
    run = wandb.init(
        project="sb3",
        entity="tjno",
        job_type="train",
        config={
            "user": "jno",
            "policy_type": "MlpPolicy",
            **params,
            "ppo_defaults": {k: str(v) for k, v in ppo_defaults.items()},
        },
        group=wandb_group(),
        sync_tensorboard=True,
        save_code=True,
    )

    env, stats_channel = make_env(time_scale=params.get("time_scale", 20.0))

    # only what params.yaml overrides; everything else = SB3 defaults
    ppo_kwargs = {
        k: v for k, v in params.items()
        if k not in ("total_timesteps", "seed", "log_interval_steps", "run_mode",
                     "eval_model_path", "eval_episodes", "time_scale")
    }
    model = PPO(
        "MlpPolicy",
        env,
        seed=params["seed"],
        verbose=0,
        tensorboard_log=f"runs/{run.id}",
        **ppo_kwargs,
    )
    run.config.update(
        {
            # actual resolved settings read from the live policy, not hardcoded
            "activation_fn": model.policy.activation_fn.__name__,
            "net_arch": str(model.policy.net_arch),
        }
    )
    callbacks = CallbackList(
        [
            TrainMetricsCallback(stats_channel, params["log_interval_steps"]),
            WandbCallback(model_save_path=f"models/{run.id}", verbose=2),
        ]
    )
    model.learn(total_timesteps=params["total_timesteps"], callback=callbacks)
    model.save("unity_model")
    upload_model(run, "unity_model")
    run.finish()
    env.close()


def run_eval(params):
    """Frozen-policy evaluation: loads the trained model and runs
    eval_episodes episodes, logging the same per-time-step metrics under
    eval/ in a separate wandb run in the same group."""
    run = wandb.init(
        project="sb3",
        entity="tjno",
        job_type="eval",
        config=params,
        group=wandb_group(),
        sync_tensorboard=False,
        save_code=True,
    )

    env, stats_channel = make_env(time_scale=params.get("time_scale", 20.0))
    model = PPO.load(params["eval_model_path"], env=env)
    model.set_random_seed(params["seed"])

    window = deque(maxlen=50)
    path_ratios, episode_rewards = [], []
    step = 0
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    def write_eval_metrics():
        n = len(path_ratios)
        metrics = {
            "eval_mean_path_ratio": sum(path_ratios) / n if n else 0.0,
            "eval_mean_episode_reward": sum(episode_rewards) / n if n else 0.0,
            "eval_success_rate_50": sum(window) / len(window) if window else 0.0,
            "eval_episodes": n,
            "total_steps": step,
        }
        metrics_path = pathlib.Path(__file__).resolve().parents[2] / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    while len(episode_rewards) < params["eval_episodes"]:
        action, _ = model.predict(obs, deterministic=True)
        step_res = env.step(action)
        if len(step_res) == 5:
            obs, reward, terminated, truncated, _info = step_res
            done = terminated or truncated
        else:
            obs, reward, done, _info = step_res
        step += 1

        if step % params["log_interval_steps"] == 0:
            wandb.log(
                {
                    "timestep": step,
                    "step_reward": float(np.mean(reward)),
                    "obs_mean": float(np.mean(obs)),
                    "obs_std": float(np.std(obs)),
                },
                step=step,
            )

        if done:
            rew = float(np.mean(reward))
            stats = stats_channel.get_and_reset_stats()
            checkpoints = stats.get("achieved_checkpoints", [(0,)])[0][0]
            path_ratio = stats.get("path_completion_ratio", [(0,)])[0][0]
            success = rew > 2.0
            window.append(1.0 if success else 0.0)
            path_ratios.append(path_ratio)
            episode_rewards.append(rew)
            wandb.log(
                {
                    "episode/success_rate": sum(window) / len(window),
                    "episode/path_ratio": path_ratio,
                    "episode/checkpoints": checkpoints,
                    "episode/reward": rew,
                },
                step=step,
            )
            write_eval_metrics()
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

    write_eval_metrics()
    run.finish()
    env.close()


def main():
    params = load_params()
    if params.get("run_mode", "train") == "eval":
        run_eval(params)
    else:
        run_train(params)


if __name__ == "__main__":
    main()
