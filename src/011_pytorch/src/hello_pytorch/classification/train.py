import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


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

    mod_0 = CircleModelV0().to(device)
    print(mod_0)
