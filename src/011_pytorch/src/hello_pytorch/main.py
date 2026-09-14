import matplotlib.pyplot as plt
import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1, dtype=torch.float, requires_grad=True)
        )
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return self.weights * x + self.bias


def main():
    weight = 0.7
    bias = 0.3

    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    torch.manual_seed(42)

    model_0 = LinearRegressionModel()

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
