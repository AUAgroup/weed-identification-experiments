from torch.utils.data import Dataset
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import AutoAugmentPolicy

from collections import Counter

from config import BATCH_SIZE, IM_SIZE
import tensorflow_datasets as tfds
from tqdm import tqdm
import torch
from skimage.transform import resize
import numpy as np


class CustomTensorDataset(Dataset):
    def __init__(self, X_tensor, y_tensor, transform_list=None):
        tensors = (X_tensor, y_tensor)
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transforms:
            # for transform in self.transforms:
            #  x = transform(x)
            x = self.transforms(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def get_transform(augmentation_mode: str):  # , preprocess):
    if augmentation_mode == "color_jitter":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
            ]
        )
    elif augmentation_mode == "color_jitter_aggressive":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7),
                transforms.ToTensor(),
            ]
        )
    elif augmentation_mode == "basic":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    elif augmentation_mode == "auto_imagenet":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
            ]
        )
    elif augmentation_mode == "auto_cifar10":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
            ]
        )
    elif augmentation_mode == "auto_SVHN":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.AutoAugment(policy=AutoAugmentPolicy.SVHN),
                transforms.ToTensor(),
            ]
        )
    elif augmentation_mode == "auto_SVHN_ImageNet":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomChoice(
                    [
                        transforms.AutoAugment(policy=AutoAugmentPolicy.SVHN),
                        transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                    ],
                    [0.5, 0.5],
                ),
                transforms.ToTensor(),
            ]
        )

    return transform


def get_raw_dataset(dataset_name: str):
    ds = tfds.load(name=dataset_name)
    data = tfds.as_numpy(ds)

    if data.get("test") is None:
        x = []
        y = []
        for ex in tqdm(data["train"]):
            x.append(ex["image"])
            y.append(ex["label"])

    return x, y


def get_dataset(
    x,
    y,
    train_indexes,
    test_indexes,
    augmentation_mode: str,  # preprocess,
    use_resampling: bool = False,
):

    transform = get_transform(augmentation_mode)  # , preprocess)

    x_train = np.array(x)[train_indexes]
    y_train = np.array(y)[train_indexes]
    x_test = np.array(x)[test_indexes]
    y_test = np.array(y)[test_indexes]
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train
    )
    # x_train, x_test, y_train, y_test = train_test_split(
    #    x, y, test_size=0.4, stratify=y
    # )
    # x_val, x_test, y_val, y_test = train_test_split(
    #    x_test, y_test, test_size=0.5, stratify=y_test
    # )

    print(Counter(y_train))
    x_train = torch.stack(
        [torch.from_numpy(resize(i, (IM_SIZE, IM_SIZE))) for i in x_train[:500]]
    )
    y_train = torch.stack([torch.from_numpy(np.array(i)) for i in y_train[:500]])
    # reshape into [C, H, W]
    x_train = x_train.swapaxes(1, 3).swapaxes(2, 3).float()
    # train_dataset = torch.utils.data.TensorDataset(x_train, y_train) [:500]

    x_val = torch.stack(
        [torch.from_numpy(resize(i, (IM_SIZE, IM_SIZE))) for i in x_val[:500]]
    )
    y_val = torch.stack([torch.from_numpy(np.array(i)) for i in y_val[:500]])
    x_val = x_val.swapaxes(1, 3).swapaxes(2, 3).float()
    # create dataset and dataloaders
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

    x_test = torch.stack(
        [torch.from_numpy(resize(i, (IM_SIZE, IM_SIZE))) for i in x_test[:500]]
    )
    y_test = torch.stack([torch.from_numpy(np.array(i)) for i in y_test[:500]])
    x_test = x_test.swapaxes(1, 3).swapaxes(2, 3).float()
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_dataset = CustomTensorDataset(x_train, y_train, transform)

    train_dataset.train = True
    val_dataset.train = False
    test_dataset.train = False

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(train_dataset, val_dataset, test_dataset):

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    return train_loader, val_loader, test_loader
