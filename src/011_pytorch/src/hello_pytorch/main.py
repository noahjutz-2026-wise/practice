import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()


def main():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    data = torch.rand(1, 3, 64, 64)
    labels = torch.rand(1, 1000)

    prediction = model(data)  # forward pass

    loss = (prediction - labels).sum()
    loss.backward()

    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optim.step()
