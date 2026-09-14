from importlib.resources import as_file, files

import torch
from torch import nn

MODEL_NAME = "01_pytorch_workflow_model_0.pt"


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
    models = files() / "models"
    with as_file(models) as m:
        m.mkdir(parents=True, exist_ok=True)
        model_save_path = m / MODEL_NAME
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

    epochs = 300

    for epoch in range(epochs):
        model_0.train()
        y_pred = model_0(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_0.eval()
        with torch.inference_mode():
            test_pred = model_0(X_test)
            test_loss = loss_fn(test_pred, y_test.type(torch.float))

            if epoch % 10 == 0:
                print(model_0.state_dict())

    torch.save(model_0.state_dict(), model_save_path)
