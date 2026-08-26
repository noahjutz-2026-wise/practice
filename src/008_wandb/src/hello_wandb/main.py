import numpy as np

import wandb

with wandb.init(
    entity="tjno",
    project="hello_wandb",
    config={
        "learning_rate": 0.02,
        "replay_size": 5e6,
        "replay_chunksize": 1024,
        "architecture": "CNN",
        "seed": 102,
    },
) as run:
    run.define_metric("step")
    run.define_metric("episode")

    # Associate metrics with their respective x-axis
    run.define_metric("step/reward", step_metric="step")
    run.define_metric("episode/total_reward", step_metric="episode")

    global_step = 0
    for episode in range(10):
        total_reward = 0

        # Per-step loop
        for env_step in range(100):
            reward = 1.0  # Example reward
            total_reward += reward

            # Log per-step metrics
            run.log({"step": global_step, "step/reward": reward})
            global_step += 1

        # Log per-episode metrics
        run.log({"episode": episode, "episode/total_reward": total_reward})


def main():
    pass
