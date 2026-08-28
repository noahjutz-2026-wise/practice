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

    n_runs = 50

    for epsilon in (0, 0.01, 0.1, 0.5, 1):
        config["epsilon"] = epsilon
        for r in range(n_runs):
            print(f"Epsilon={epsilon}; Run={r}")
            with wandb.init(
                mode="online",
                entity="tjno",
                project="monte_carlo",
                config=config,
                group="offpolicy_3",
            ) as run:
                run.define_metric("episode")
                run.define_metric("cum_reward", step_metric="episode", summary="max")
                run.define_metric("steps", step_metric="episode", summary="max")
                run.define_metric("q_coverage", step_metric="episode", summary="max")
                run.define_metric("q_value", step_metric="episode", summary="mean")
                run.define_metric("stability", step_metric="episode")

                Q = train.train(run)
