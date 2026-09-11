import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule

from hello_torchrl.environment.flappy_bird import (
    get_env,
    get_env_dreamer,
    get_env_human,
)

WANDB_API_KEY = os.environ["WANDB_API_KEY"]
RAY_TEMP_DIR = Path(os.environ["RAY_TEMP_DIR"])
RAY_ADDRESS = os.environ["RAY_ADDRESS"]

# Create only the neural network (RLModule) from our algorithm checkpoint.
# See here (https://docs.ray.io/en/master/rllib/checkpoints.html)
# to learn more about checkpointing and the specific "path" used.
rl_module = RLModule.from_checkpoint(
    Path(RAY_TEMP_DIR)
    / "ray_results"
    / "DreamerV3_2026-09-10_07-36-02"
    / "DreamerV3_flappy-bird_f282c_00000_0_2026-09-10_07-36-02/"
    / "checkpoint_000091"
    / "learner_group"
    / "learner"
    / "rl_module"
    / "default_policy"
)

# Create the RL environment to test against (same as was used for training earlier).
env = get_env_human()

episode_return = 0.0
done = False

# Reset the env to get the initial observation.
obs, info = env.reset()

# Initialize recurrent state for DreamerV3
state = rl_module.get_initial_state()
batched_state = {k: v.unsqueeze(0) for k, v in state.items()}
is_first = torch.tensor([1.0], dtype=torch.float32)

while not done:
    # Uncomment this line to render the env.
    # env.render()

    # Compute the next action from a batch (B=1, T=1) of observations.
    obs_batch = (
        torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)
    )  # add B=1 and T=1 dimensions

    batch = {
        "obs": obs_batch,
        "state_in": batched_state,
        "is_first": is_first,
    }

    model_outputs = rl_module.forward_inference(batch)

    # Extract the discrete action from the output (B=1, T=1)
    greedy_action = model_outputs["actions"][0, 0].item()

    # Update state and is_first for the next step
    batched_state = model_outputs["state_out"]
    is_first = torch.tensor([0.0], dtype=torch.float32)

    # Send the action to the environment for the next step.
    obs, reward, terminated, truncated, info = env.step(greedy_action)

    # Perform env-loop bookkeeping.
    episode_return += reward
    done = terminated or truncated

print(f"Reached episode return of {episode_return}.")
