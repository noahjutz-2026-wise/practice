import wandb

project = "my-awesome-proj"

with wandb.init(project=project) as run:
    for i in range(1000):
        run.log({"accuracy": i, "loss": -float(i) / 10})


def main():
    pass
