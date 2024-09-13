import os
import pickle
import string

import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset

import numpy as np
from PIL import Image


class SubFEMNIST(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.

    Attributes
    ----------
    transform
    data: iterable of integers
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__

    """
    def __init__(self, path):
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        self.data, self.targets = torch.load(path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = np.uint8(img.numpy() * 255)
        img = Image.fromarray(img, mode='L').resize((32, 32)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SubCIFAR10(Dataset):
    """
    Constructs a subset of CIFAR10 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__

    """

    def __init__(self, path, aggregator_, cifar10_data=None, cifar10_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param cifar10_data: Cifar-10 dataset inputs stored as torch.tensor
        :param cifar10_targets: Cifar-10 dataset labels stored as torch.tensor
        :param transform:

        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        self.aggregator_ = aggregator_

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])
        if self.aggregator_ == "centralized":
            if cifar10_data is None or cifar10_targets is None:
                self.data, self.targets = get_cifar10()
            else:
                self.data, self.targets = cifar10_data, cifar10_targets
        else:
            # Convert input embeddings/targets from NumPy arrays to PyTorch tensors
            if cifar10_data is not None and cifar10_targets is not None:
                self.data = torch.tensor(cifar10_data, dtype=torch.float32)  
                self.targets = torch.tensor(cifar10_targets, dtype=torch.int64)  
            else:
                # Load the data from the path if not provided (this part is based on your original implementation)
                raise NotImplementedError("Loading from file path is not implemented in this example.")

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        if self.aggregator_ == "centralized":
            img, target = self.data[index], self.targets[index]

            img = Image.fromarray(img.numpy())

            if self.transform is not None:
                img = self.transform(img)

            target = target

            return img, target, index
        
        else:
            # Directly return the data (embedding) and the corresponding target
            return self.data[index], self.targets[index], index


class SubCIFAR100(Dataset):
    """
    Constructs a subset of CIFAR100 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__

    """
    def __init__(self, path, aggregator_, cifar100_data=None, cifar100_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices:
        :param cifar100_data: CIFAR-100 dataset inputs
        :param cifar100_targets: CIFAR-100 dataset labels
        :param transform:

        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])
        
        self.aggregator_ = aggregator_

        if self.aggregator_ == "centralized":
            if cifar100_data is None or cifar100_targets is None:
                self.data, self.targets = get_cifar100()
            else:
                self.data, self.targets = cifar100_data, cifar100_targets
        else:
            # Convert input embeddings/targets from NumPy arrays to PyTorch tensors
            if cifar100_data is not None and cifar100_targets is not None:
                self.data = torch.tensor(cifar100_data, dtype=torch.float32)  
                self.targets = torch.tensor(cifar100_targets, dtype=torch.int64)  
            else:
                # Load the data from the path if not provided (this part is based on your original implementation)
                raise NotImplementedError("Loading from file path is not implemented in this example.")

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        if self.aggregator_ == "centralized":
            img, target = self.data[index], self.targets[index]

            img = Image.fromarray(img.numpy())

            if self.transform is not None:
                img = self.transform(img)

            target = target

            return img, target, index
        
        else:
            # Directly return the data (embedding) and the corresponding target
            return self.data[index], self.targets[index], index
    

def get_cifar10():
    """
    gets full (both train and test) CIFAR10 dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)

    :return:
        cifar10_data, cifar10_targets

    """
    cifar10_path = os.path.join("data", "cifar10", "dataset")
    assert os.path.isdir(cifar10_path), "Download cifar10 dataset!!"

    cifar10_train =\
        CIFAR10(
            root=cifar10_path,
            train=True, download=False
        )

    cifar10_test =\
        CIFAR10(
            root=cifar10_path,
            train=False,
            download=False)

    cifar10_data = \
        torch.cat([
            torch.tensor(cifar10_train.data),
            torch.tensor(cifar10_test.data)
        ])

    cifar10_targets = \
        torch.cat([
            torch.tensor(cifar10_train.targets),
            torch.tensor(cifar10_test.targets)
        ])

    return cifar10_data, cifar10_targets


def get_cifar100():
    """
    gets full (both train and test) CIFAR100 dataset inputs and labels;
    the dataset should be first downloaded (see data/cifar100/README.md)

    :return:
        cifar100_data, cifar100_targets

    """
    cifar100_path = os.path.join("data", "cifar100", "dataset")
    assert os.path.isdir(cifar100_path), "Download cifar100 dataset!!"

    cifar100_train =\
        CIFAR100(
            root=cifar100_path,
            train=True, download=False
        )

    cifar100_test =\
        CIFAR100(
            root=cifar100_path,
            train=False,
            download=False)

    cifar100_data = \
        torch.cat([
            torch.tensor(cifar100_train.data),
            torch.tensor(cifar100_test.data)
        ])

    cifar100_targets = \
        torch.cat([
            torch.tensor(cifar100_train.targets),
            torch.tensor(cifar100_test.targets)
        ])

    return cifar100_data, cifar100_targets
