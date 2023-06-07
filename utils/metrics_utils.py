from torchmetrics.classification import Accuracy
from torchmetrics import F1Score
from torchmetrics import ConfusionMatrix
import torch

# Check accuracy on training & test to see how good our model
def check_performances(loader, model, num_classes, preprocess, device):
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    confmat = ConfusionMatrix(
        task="multiclass", num_classes=num_classes, normalize="true"
    ).to(device)

    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = preprocess(x)
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            batch_acc = accuracy(scores, y)
            batch_f1 = f1(scores, y)
            batch_confmat = confmat(scores, y)

    model.train()

    return accuracy.compute().item(), f1.compute().item(), confmat.compute()


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model, num_classes, preprocess, device):
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = preprocess(x)
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            batch_acc = accuracy(scores, y)

    model.train()

    return accuracy.compute().item()
