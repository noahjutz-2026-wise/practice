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
        _ = train.train(run, Q)


def experiment_train(config: wandb.sdk.Config, model_name: str | None = None):
    config["task"] = "train"
    n_runs = 1

    for r in range(n_runs):
        print(f"Run={r}")
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

            Q = train.train(run)

            if model_name is not None:
                np.savez_compressed("model.npz", Q)
                run.log_model(path="model.npz", name=model_name)


def main():
    config = {
        "n_bins": (15, 15, 15, 15),
        "gamma": 0.9,
        "epsilon": 0.1,
        "episodes": 20000,
        "log_every": 100,
    }

    experiment_train(config, model_name="mc_offpolicy")
