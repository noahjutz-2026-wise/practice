import numpy as np

import wandb

from . import train

entity = "tjno"
project = "monte_carlo"
mode = "online"
group = None


def experiment_eval(config: wandb.sdk.Config):
    config["task"] = "eval"
    with wandb.init(
        mode=mode, entity=entity, project=project, config=config, group=group
    ) as run:
        p = run.use_model(name="mc_offpolicy:v0")
        Q = np.load(p)
        _ = train(run, Q)


def experiment_train(config: wandb.sdk.Config):
    config["task"] = "train"
    n_runs = 50

    for epsilon in (0, 0.01, 0.1, 0.5, 1):
        config["epsilon"] = epsilon
        for r in range(n_runs):
            print(f"Epsilon={epsilon}; Run={r}")
            with wandb.init(
                mode=mode,
                entity=entity,
                project=project,
                config=config,
                group=group,
            ) as run:
                run.define_metric("episode")
                run.define_metric("cum_reward", step_metric="episode", summary="max")
                run.define_metric("steps", step_metric="episode", summary="max")
                run.define_metric("q_coverage", step_metric="episode", summary="max")
                run.define_metric("q_value", step_metric="episode", summary="mean")
                run.define_metric("stability", step_metric="episode")

                _ = experiment_train.train(run)


def main():
    config = {
        "n_bins": (15, 15, 15, 15),
        "gamma": 0.9,
        "epsilon": 0.01,
        "episodes": 2000,
        "log_every": 100,
    }

    experiment_eval(config)
