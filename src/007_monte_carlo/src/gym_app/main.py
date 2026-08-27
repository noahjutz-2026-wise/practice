import wandb

from . import train


def main():
    config = {
        "n_bins": (15, 15, 15, 15),
        "gamma": 0.9,
        "epsilon": 0.01,
        "episodes": 2000,
    }

    with wandb.init(entity="jno", project="monte_carlo", config=config) as run:
        train.train(run)
