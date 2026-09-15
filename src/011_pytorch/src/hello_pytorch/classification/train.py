import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

from hello_pytorch.helper_functions import plot_decision_boundary, plot_predictions

device = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_pred)


class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))


def main():
    n_samples = 1000
    X, y = make_circles(n_samples, noise=0.03, random_state=42)

    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_test, y_train, y_test = (
        X_train.to(device),
        X_test.to(device),
        y_train.to(device),
        y_test.to(device),
    )

    mod_0 = nn.Sequential(
        nn.Linear(in_features=2, out_features=5),
        nn.Linear(in_features=5, out_features=1),
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=mod_0.parameters(), lr=0.1)

    epochs = 100

    for epoch in range(epochs):
        mod_0.train()
        y_logits = mod_0(X_train).squeeze()
        y_pred = torch.sigmoid(y_logits)
        y_pred = torch.round(y_pred)

        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mod_0.eval()
        with torch.inference_mode():
            y_logits_test = mod_0(X_test).squeeze()
            y_pred_test = torch.sigmoid(y_logits_test)
            y_pred_test = torch.round(y_logits_test)

            loss_test = loss_fn(y_logits_test, y_test)
            acc_test = accuracy_fn(y_test, y_pred_test)

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {loss_test:.5f}, Test acc: {acc_test:.2f}%"
                )
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(mod_0, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(mod_0, X_test, y_test)
    plt.show()
