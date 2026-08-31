import numpy as np

import wandb

from . import train

entity = "tjno"
project = "rl"
mode = "online"
group = "compare_mc_ql_sarsa_expected_sarsa_1"


def experiment_eval(config: wandb.sdk.Config, model_name: str | None = None):
    config["task"] = "eval"
    with wandb.init(
        mode=mode, entity=entity, project=project, config=config, group=group
    ) as run:
        p = run.use_model(name=model_name)
        Q = np.load(p)["arr_0"]
        _ = train.train(run, Q)


def experiment_train(config: wandb.sdk.Config, model_name: str | None = None):
    config["task"] = "train"
    n_runs = 500

    for r in range(n_runs):
        print(f"Run={r}")
        for method in ["expected_sarsa", "sarsa", "q_learning", "monte_carlo"]:
            print(f"Run {r} method {method}")
            config["prediction_method"] = method
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
        "prediction_method": "expected_sarsa",
        "n_bins": (15, 15, 15, 15),
        "gamma": 0.9,  # Discount factor
        "epsilon": 0.1,  # Exploration rate
        "alpha": 0.05,  # Step size
        "episodes": 2000,
        "log_every": 100,
    }

    experiment_train(config)
