from datetime import UTC, datetime

import numpy as np

import wandb

from . import train


def main():
    config = {
        "task": "train",
        "n_bins": (15, 15, 15, 15),
        "gamma": 0.9,
        "epsilon": 0.01,
        "episodes": 2000,
        "log_every": 100,
    }

    with wandb.init(entity="tjno", project="monte_carlo", config=config) as run:
        run.define_metric("episode")
        run.define_metric("cum_reward", step_metric="episode", summary="max")
        Q = train.train(run)
        np.savez_compressed("model.npz", Q)
        name = f"mc_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        run.log_model(path="model.npz", name=name)
