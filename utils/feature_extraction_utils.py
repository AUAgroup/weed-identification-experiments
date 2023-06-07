import torch
from torch import nn
from tqdm import tqdm
import numpy as np


def extract_features(
    model, model_name, train_loader, val_loader, test_loader, preprocess, device
):
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    model.eval()

    if model_name.startswith("trans"):
        model.head = nn.Sequential()
    elif model_name.startswith("regnet"):
        model.fc = nn.Sequential()
    else:
        model.classifier = nn.Sequential()

    with torch.no_grad():

        for data, targets in tqdm(train_loader):
            data = preprocess(data)
            data = data.to(device=device)
            features = model(data)
            train_features.extend(features.cpu().numpy())
            train_labels.extend(targets.numpy())

        for data, targets in tqdm(test_loader):
            data = preprocess(data)
            data = data.to(device=device)
            features = model(data)
            test_features.extend(features.cpu().numpy())
            test_labels.extend(targets.numpy())

        for data, targets in tqdm(val_loader):
            data = preprocess(data)
            data = data.to(device=device)
            features = model(data)
            test_features.extend(features.cpu().numpy())
            test_labels.extend(targets.numpy())

    return (
        np.array(train_features),
        np.array(train_labels),
        np.array(test_features),
        np.array(test_labels),
    )
