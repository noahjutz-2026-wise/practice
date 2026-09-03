import torch
from tensordict import TensorDict


def main():
    td = TensorDict(
        {
            "z": torch.zeros((10, 10)),
            "o": torch.ones((10, 10, 2)),
            "r": torch.rand((10, 10, 2, 20)),
        },
        batch_size=[10, 10],
    )
    print(td[..., :2])
