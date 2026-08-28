import wandb

from . import train


def main():
    config = {
        "task": "train",
        "n_bins": (15, 15, 15, 15),
        "gamma": 0.9,
        "epsilon": 0.01,
        "episodes": 1000,
        "log_every": 100,
    }

    n_runs = 50

    for r in range(n_runs):
        print(f"Run {r}")
        with wandb.init(
            mode="disabled",
            entity="tjno",
            project="monte_carlo",
            config=config,
            group="offpolicy_1",
        ) as run:
            run.define_metric("episode")
            run.define_metric("cum_reward", step_metric="episode", summary="max")
            run.define_metric("steps", step_metric="episode", summary="max")
            run.define_metric("q_coverage", step_metric="episode", summary="max")
            run.define_metric("q_value", step_metric="episode", summary="mean")
            run.define_metric("stability", step_metric="episode")

            Q = train.train(run)
