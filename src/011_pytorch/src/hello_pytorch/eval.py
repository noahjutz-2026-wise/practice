from importlib.resources import as_file, files

import torch

from hello_pytorch.main import LinearRegressionModel

MODEL_NAME = "01_pytorch_workflow_model_0.pt"


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

    loaded_model_0 = LinearRegressionModel()
    loaded_model_0.load_state_dict(torch.load(model_save_path))

    loaded_model_0.eval()
    with torch.inference_mode():
        preds = loaded_model_0(X_test)
        print(preds - y_test)
