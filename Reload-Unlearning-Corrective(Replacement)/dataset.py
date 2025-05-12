"""
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
"""
import copy
import glob
import os
from shutil import move

import numpy as np
import torch
from PIL import Image
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder, MNIST, USPS
from tqdm import tqdm

import ucimlrepo

import sklearn


def cifar10_dataloaders_no_val(
        batch_size=128, data_dir="datasets/cifar10", num_workers=2
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    val_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def svhn_dataloaders(
        batch_size=128,
        data_dir="datasets/svhn",
        num_workers=2,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
):
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: SVHN\t 45000 images for training \t 5000 images for validation\t"
    )

    train_set = SVHN(data_dir, split="train", transform=train_transform, download=True)

    test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

    train_set.labels = np.array(train_set.labels)
    test_set.labels = np.array(test_set.labels)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.labels) + 1):
        class_idx = np.where(train_set.labels == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.labels = train_set_copy.labels[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.labels = train_set_copy.labels[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 4454:
            test_set.data = test_set.data[test_set.labels != class_to_replace]
            test_set.labels = test_set.labels[test_set.labels != class_to_replace]

    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(
        batch_size=128,
        data_dir="datasets/cifar100",
        num_workers=2,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")
    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None or indexes_to_replace == 450:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders_no_val(
        batch_size=128, data_dir="datasets/cifar100", num_workers=2
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    val_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

class SyntheticDataset(torch.utils.data.TensorDataset):
    def __init__(self, features, labels):
        super(SyntheticDataset, self).__init__(features, labels)
        self.data = features
        self.targets = labels

    def __len__(self):
        return len(self.data)


def generate_challenging_dataset_v3(n_samples, n_features=2, boundary_proportion=0.2, blob_proportion=0.4):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    num_boundary_points = int(boundary_proportion * n_samples)
    num_blob_points = int(blob_proportion * n_samples)
    num_xor_points = n_samples - num_boundary_points - num_blob_points

    # First 20% points for the decision boundary (randomly assigned classes)
    for i in range(num_boundary_points):
        X[i] = np.random.uniform(-10, 10, size=n_features)
        y[i] = np.random.choice([0, 1])

    # Next 40% points for Gaussian blobs
    centers = [(-5, -5), (5, 5)]
    cluster_std = 1.5
    X_blobs, y_blobs = make_blobs(n_samples=num_blob_points, centers=centers, cluster_std=cluster_std, random_state=42)

    X[num_boundary_points:num_boundary_points + num_blob_points] = X_blobs
    y[num_boundary_points:num_boundary_points + num_blob_points] = y_blobs

    # Last 40% points for XOR pattern
    for i in range(num_boundary_points + num_blob_points, n_samples):
        x1 = np.random.uniform(-10, 10)
        x2 = np.random.uniform(-10, 10)
        X[i] = [x1, x2]
        y[i] = 1 if (x1 > 0) ^ (x2 > 0) else 0

    X, y = shuffle(X, y)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_challenging_dataset_v2(n_samples, n_features=2, boundary_proportion=0.1, circle_proportion=0.4):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    num_boundary_points = int(boundary_proportion * n_samples)
    num_circle_points = int(circle_proportion * n_samples)
    num_sinusoid_points = n_samples - num_boundary_points - num_circle_points

    # First 20% points for the decision boundary (randomly assigned classes)
    for i in range(num_boundary_points):
        X[i] = np.random.uniform(-10, 10, size=n_features)
        y[i] = np.random.choice([0, 1])

    # Next 40% points for concentric circles
    radius_inner = 5
    radius_outer = 10
    theta = np.linspace(0, 2 * np.pi, num_circle_points // 2)
    for i in range(num_boundary_points, num_boundary_points + num_circle_points // 2):
        X[i] = radius_inner * np.array([np.cos(theta[i - num_boundary_points]), np.sin(theta[i - num_boundary_points])])
        y[i] = 0

    for i in range(num_boundary_points + num_circle_points // 2, num_boundary_points + num_circle_points):
        X[i] = radius_outer * np.array([np.cos(theta[i - num_boundary_points - num_circle_points // 2]),
                                        np.sin(theta[i - num_boundary_points - num_circle_points // 2])])
        y[i] = 1

    # Last 40% points based on a sinusoidal curve
    for i in range(num_boundary_points + num_circle_points, n_samples):
        x = np.random.uniform(-10, 10)
        y_temp = 5 * np.sin(x)
        X[i] = [x, y_temp]
        y[i] = 1 if y_temp > 0 else 0

    X, y = shuffle(X, y)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_challenging_dataset(n_samples, n_features, boundary_proportion=0.16, noise_proportion=0.16):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    num_boundary_points = int(boundary_proportion * n_samples)
    num_noisy_points = int(noise_proportion * n_samples)
    num_class_points = n_samples - num_boundary_points - num_noisy_points

    # First 20% points for the decision boundary (randomly assigned classes)
    for i in range(num_boundary_points):
        X[i] = np.random.uniform(-10, 10, size=n_features)
        y[i] = np.random.choice([0, 1])

    # Next 40% points for clear class separation
    for i in range(num_boundary_points, num_boundary_points + num_class_points // 2):
        X[i] = np.random.normal(loc=-5, scale=1, size=n_features)
        y[i] = 0

    for i in range(num_boundary_points + num_class_points // 2, num_boundary_points + num_class_points):
        X[i] = np.random.normal(loc=5, scale=1, size=n_features)
        y[i] = 1

    # Last 40% points with noise and overlap
    for i in range(num_boundary_points + num_class_points, n_samples):
        if np.random.uniform() < 0.5:
            X[i] = np.random.normal(loc=0, scale=5, size=n_features)
            y[i] = 0
        else:
            X[i] = np.random.normal(loc=0, scale=5, size=n_features)
            y[i] = 1

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_noisy_forget_synthetic(n_samples, n_features, boundary_proportion=0.2):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    num_boundary_points = int(boundary_proportion * n_samples)
    num_remaining_points = n_samples - num_boundary_points
    num_class_points = num_remaining_points // 2

    # First 20% points for the decision boundary (randomly assigned classes)
    for i in range(num_boundary_points):
        X[i] = np.random.uniform(-10, 10, size=n_features)
        y[i] = np.random.choice([0, 1])

    # Next 40% points for class 0
    for i in range(num_boundary_points, num_boundary_points + num_class_points):
        X[i] = np.random.normal(loc=-2, scale=2, size=n_features)
        y[i] = 0

    # Next 40% points for class 1
    for i in range(num_boundary_points + num_class_points, n_samples):
        X[i] = np.random.normal(loc=2, scale=2, size=n_features)
        y[i] = 1

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_unlearning_resistant_dataset(n_samples, n_features=2, structured_noise_proportion=0.1,
                                          total_noise_proportion=0.5):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    num_structured_noise_points = int(structured_noise_proportion * n_samples)
    num_total_noise_points = int(total_noise_proportion * n_samples)
    num_different_structure_points = n_samples - num_total_noise_points

    # First 10% points (structured noise)
    for i in range(num_structured_noise_points):
        x1 = np.random.uniform(-10, 10)
        x2 = np.sin(x1) + np.random.normal(0, 0.5)
        X[i] = [x1, x2]
        y[i] = 1 if i % 2 == 0 else 0

    # Next 40% points (remaining noise)
    for i in range(num_structured_noise_points, num_total_noise_points):
        x1 = np.random.uniform(-10, 10)
        x2 = np.random.normal(0, 5)
        X[i] = [x1, x2]
        y[i] = 1 if i % 2 == 0 else 0

    # Last 50% points (drastically different structure)
    for i in range(num_total_noise_points, n_samples):
        x1 = np.random.uniform(-10, 10)
        x2 = x1 * 0.5 + np.random.normal(0, 1)
        X[i] = [x1, x2]
        y[i] = 1 if i % 2 == 0 else 0

    X, y = shuffle(X, y)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_n_grid_gaussian_blobs(n_samples, n_features, n_classes=2, grid_size=3):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    x_max = grid_size * 4
    x_min = -x_max
    y_max = grid_size * 4
    y_min = -y_max

    centers = []
    for i in range(grid_size):
        for j in range(grid_size):
            centers.append((x_min + (x_max - x_min) * i / grid_size, y_min + (y_max - y_min) * j / grid_size))

    # Shuffle the centers to avoid any pattern in the data
    centers = np.array(centers)

    # Set the seed to ensure reproducibility
    np.random.seed(2)

    np.random.shuffle(centers)

    cluster_std = 1.5

    target_class = 0
    start_idx = 0

    # Add noise around the border of the blobs
    num_noise = int(0.2 * n_samples)
    # X_noise = np.random.uniform(low=[x_min * 1.25, y_min * 1.25], high=[x_max * 1.25, y_max * 1.25], size=(num_noise, n_features))
    X_noise = np.random.normal(loc=0, scale=5, size=(num_noise, n_features))
    y_noise = np.random.choice([0, 1], size=num_noise)

    X[start_idx:start_idx + num_noise] = X_noise
    y[start_idx:start_idx + num_noise] = y_noise

    start_idx += num_noise

    num_points_per_blob = (n_samples - num_noise) // (len(centers))

    for center in centers:
        # Generate num_points_per_blob points for each blob around the center
        X_temp = np.random.normal(loc=center, scale=cluster_std, size=(num_points_per_blob, n_features))
        X[start_idx:start_idx + num_points_per_blob] = X_temp
        y[start_idx:start_idx + num_points_per_blob] = target_class
        target_class = 1 - target_class
        start_idx += num_points_per_blob

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_unlearning_challenging_dataset(n_samples, n_features=2, influential_proportion=0.1):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    num_influential_points = int(influential_proportion * n_samples)
    num_remaining_points = n_samples - num_influential_points

    # First 10% points (highly influential outliers)
    for i in range(num_influential_points):
        x1 = np.random.uniform(-50, 50)
        x2 = x1 + np.random.normal(0, 15)  # Slightly perturbed linear relationship
        X[i] = [x1, x2]
        y[i] = 1 if x1 + x2 > 0 else 0

    # Add more noise to the influential points to make them crucial for the boundary
    noise_indices = np.random.choice(num_influential_points, size=int(0.3 * num_influential_points), replace=False)
    for idx in noise_indices:
        X[idx] += np.random.normal(0, 5, size=n_features)
        y[idx] = 1 - y[idx]

    # Next 90% points (normally distributed)
    for i in range(num_influential_points, n_samples):
        x1 = np.random.normal(0, 5)
        x2 = x1 + np.random.normal(0, 1)  # Slightly perturbed line y = x
        X[i] = [x1, x2]
        y[i] = 1 if x1 + x2 > 0 else 0

    X, y = shuffle(X, y)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_boundary_sensitive_dataset(n_samples, n_features=2, key_proportion=0.18):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    num_key_points = int(key_proportion * n_samples)
    num_remaining_points = n_samples - num_key_points

    # First 10% points (key points that affect the decision boundary)
    for i in range(num_key_points):
        x1 = np.random.uniform(-5, 5)
        x2 = np.random.uniform(-5, 5)
        X[i] = [x1, x2]
        y[i] = 1 if x1 + x2 > 0 else 0

    # Next 90% points (linearly separable)
    for i in range(num_key_points, n_samples):
        x1 = np.random.uniform(-10, 10)
        x2 = x1 + np.random.normal(0, 1)  # Slightly perturbed line y = x
        X[i] = [x1, x2]
        y[i] = 1 if x1 > 0 else 0

    # Add some noise to the key points to make them impactful
    noise_indices = np.random.choice(num_key_points, size=int(0.2 * num_key_points), replace=False)
    for idx in noise_indices:
        X[idx] += np.random.normal(0, 5, size=n_features)
        y[idx] = 1 - y[idx]

    # X, y = shuffle(X, y)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_triple_boundary_dataset(n_samples, n_features=2, boundary1_proportion=0.3, boundary2_proportion=0.3):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    num_boundary1_points = int(boundary1_proportion * n_samples)
    num_boundary2_points = int(boundary2_proportion * n_samples)
    num_boundary3_points = n_samples - num_boundary1_points - num_boundary2_points

    # First 20% points for the first non-linear boundary (circular)
    for i in range(num_boundary1_points // 2):
        r = np.random.uniform(0, 5)
        theta = np.random.uniform(0, 2 * np.pi)
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        X[i] = [x1, x2]
        y[i] = 0

    for i in range(num_boundary1_points // 2, num_boundary1_points):
        r = np.random.uniform(5, 10)
        theta = np.random.uniform(0, 2 * np.pi)
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        X[i] = [x1, x2]
        y[i] = 1

    # Next 30% points for the second non-linear boundary (sine wave)
    for i in range(num_boundary1_points, num_boundary1_points + num_boundary2_points // 2):
        x1 = np.random.uniform(-10, 10)
        x2 = np.sin(x1) + np.random.normal(0, 0.5)
        X[i] = [x1, x2]
        y[i] = 0

    for i in range(num_boundary1_points + num_boundary2_points // 2, num_boundary1_points + num_boundary2_points):
        x1 = np.random.uniform(-10, 10)
        x2 = np.sin(x1 + np.pi) + np.random.normal(0, 0.5)
        X[i] = [x1, x2]
        y[i] = 1

    # Remaining 50% points for the third non-linear boundary (parabolic)
    for i in range(num_boundary1_points + num_boundary2_points,
                   num_boundary1_points + num_boundary2_points + num_boundary3_points // 2):
        x1 = np.random.uniform(-10, 10)
        x2 = 0.1 * x1 ** 2 + np.random.normal(0, 1)
        X[i] = [x1, x2]
        y[i] = 0

    for i in range(num_boundary1_points + num_boundary2_points + num_boundary3_points // 2, n_samples):
        x1 = np.random.uniform(-10, 10)
        x2 = -0.1 * x1 ** 2 + np.random.normal(0, 1)
        X[i] = [x1, x2]
        y[i] = 1

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_multiclass_unlearning_challenging_dataset(n_samples, n_classes, n_features=2):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    segment_size = n_samples // n_classes

    # Define non-linear boundaries for each class
    def nonlinear_boundary_1(x):
        return np.sin(x) + np.random.normal(0, 0.5)

    def nonlinear_boundary_2(x):
        return np.cos(x) + np.random.normal(0, 0.5)

    def nonlinear_boundary_3(x):
        return 0.1 * x ** 2 + np.random.normal(0, 0.5)

    def nonlinear_boundary_4(x):
        return -0.1 * x ** 2 + np.random.normal(0, 0.5)

    boundaries = [nonlinear_boundary_1, nonlinear_boundary_2, nonlinear_boundary_3, nonlinear_boundary_4]

    for class_idx in range(n_classes):
        boundary_func = boundaries[class_idx % len(boundaries)]

        for i in range(segment_size):
            x1 = np.random.uniform(-10, 10)
            if i % 2 == 0:
                x2 = boundary_func(x1)
                y[class_idx * segment_size + i] = class_idx
            else:
                x2 = np.random.normal(0, 5)
                y[class_idx * segment_size + i] = np.random.choice(n_classes)
            X[class_idx * segment_size + i] = [x1, x2]

    X, y = shuffle(X, y)
    X = X.astype(np.float32)

    return X, y


def generate_hard_to_unlearn_dataset(n_samples, n_features=2):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    # Generate non-linear boundaries
    def nonlinear_boundary_1(x):
        return np.sin(x) + np.random.normal(0, 0.5)

    def nonlinear_boundary_2(x):
        return np.cos(x) + np.random.normal(0, 0.5)

    def nonlinear_boundary_3(x):
        return 0.1 * x ** 2 + np.random.normal(0, 0.5)

    def nonlinear_boundary_4(x):
        return -0.1 * x ** 2 + np.random.normal(0, 0.5)

    # Number of points per segment
    segment_size = n_samples // 4

    # First segment: Boundary 1 and noise
    for i in range(segment_size):
        x1 = np.random.uniform(-10, 10)
        if i % 2 == 0:
            x2 = nonlinear_boundary_1(x1)
            y[i] = 0
        else:
            x2 = np.random.normal(0, 5)
            y[i] = 1
        X[i] = [x1, x2]

    # Second segment: Boundary 2 and noise
    for i in range(segment_size, 2 * segment_size):
        x1 = np.random.uniform(-10, 10)
        if i % 2 == 0:
            x2 = nonlinear_boundary_2(x1)
            y[i] = 1
        else:
            x2 = np.random.normal(0, 5)
            y[i] = 0
        X[i] = [x1, x2]

    # Third segment: Boundary 3 and noise
    for i in range(2 * segment_size, 3 * segment_size):
        x1 = np.random.uniform(-10, 10)
        if i % 2 == 0:
            x2 = nonlinear_boundary_3(x1)
            y[i] = 0
        else:
            x2 = np.random.normal(0, 5)
            y[i] = 1
        X[i] = [x1, x2]

    # Fourth segment: Boundary 4 and noise
    for i in range(3 * segment_size, n_samples):
        x1 = np.random.uniform(-10, 10)
        if i % 2 == 0:
            x2 = nonlinear_boundary_4(x1)
            y[i] = 1
        else:
            x2 = np.random.normal(0, 5)
            y[i] = 0
        X[i] = [x1, x2]

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_dual_boundary_dataset(n_samples, n_features=2, boundary_proportion=0.2):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    num_boundary1_points = int(boundary_proportion * n_samples)
    num_boundary2_points = n_samples - num_boundary1_points

    # First 30% points for the first decision boundary
    for i in range(num_boundary1_points // 2):
        x1 = np.random.uniform(-10, 0)
        x2 = 2 * x1 + np.random.normal(0, 1)
        X[i] = [x1, x2]
        y[i] = 0

    for i in range(num_boundary1_points // 2, num_boundary1_points):
        x1 = np.random.uniform(0, 10)
        x2 = -2 * x1 + np.random.normal(0, 1)
        X[i] = [x1, x2]
        y[i] = 1

    # Remaining 70% points for the second decision boundary
    for i in range(num_boundary1_points, num_boundary1_points + num_boundary2_points // 2):
        x1 = np.random.uniform(-10, 0)
        x2 = 0.5 * x1 + np.random.normal(0, 1)
        X[i] = [x1, x2]
        y[i] = 1

    for i in range(num_boundary1_points + num_boundary2_points // 2, n_samples):
        x1 = np.random.uniform(0, 10)
        x2 = -0.5 * x1 + np.random.normal(0, 1)
        X[i] = [x1, x2]
        y[i] = 0

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_duplicate_dataset(n_samples, n_features=2, duplicate_proportion=0.2, noise=0.0):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    num_duplicate_points = int(duplicate_proportion * n_samples)
    num_remaining_points = n_samples - 2 * num_duplicate_points

    # Use another generation mechanism to generate the duplicate points
    X_duplicate, y_duplicate = generate_unlearning_challenging_dataset(num_duplicate_points, n_features)

    X[:num_duplicate_points] = X_duplicate
    y[:num_duplicate_points] = y_duplicate

    # Copy the duplicate points to the second half of the dataset
    X[num_duplicate_points:2 * num_duplicate_points] = X_duplicate
    y[num_duplicate_points:2 * num_duplicate_points] = np.ones_like(y_duplicate) - y_duplicate  # Flip the labels
    # y[num_duplicate_points:2 * num_duplicate_points] = y_duplicate

    # Add noise to the duplicate points
    if noise > 0:
        X[:num_duplicate_points] += np.random.normal(0, noise, size=(num_duplicate_points, n_features))

    # Generate the remaining points using another generation mechanism
    X_remaining, y_remaining = generate_triple_boundary_dataset(num_remaining_points, n_features)

    X[2 * num_duplicate_points:] = X_remaining
    y[2 * num_duplicate_points:] = y_remaining

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_noise_dataset(n_samples, n_features=2, type="uniform"):
    # Generate n_samples points with noise
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    if type == "uniform":
        X = np.random.uniform(-10, 10, size=(n_samples, n_features))
    elif type == "normal":
        X = np.random.normal(0, 5, size=(n_samples, n_features))
    else:
        raise ValueError("Invalid noise type")

    y = np.random.choice([0, 1], size=n_samples)

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


# Generate all duplicate
def generate_all_duplicate_dataset(n_samples, n_features=2, duplicate_proportion=0.2, noise=0.0):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    num_duplicate_points = int(duplicate_proportion * n_samples)

    # Use another generation mechanism to generate the duplicate points
    X_duplicate, y_duplicate = generate_challenging_dataset(num_duplicate_points, n_features)

    X[:num_duplicate_points] = X_duplicate
    y[:num_duplicate_points] = y_duplicate

    # Add noise to the duplicate points
    if noise > 0:
        X[:num_duplicate_points] += np.random.normal(0, noise, size=(num_duplicate_points, n_features))

    # Replicate the duplicate points to the remaining points
    multiplier = (1 / duplicate_proportion) - 1
    X[num_duplicate_points:] = np.tile(X_duplicate, (int(multiplier), 1))
    y[num_duplicate_points:] = np.tile(y_duplicate, int(multiplier))

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_all_duplicate_classes_dataset(n_samples, n_features=2, n_classes=2, duped_classes=2, noise=0.0):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    points_per_class = n_samples // n_classes

    X_cls, y_cls = generate_challenging_dataset(points_per_class, n_features)

    for cls in range(duped_classes):
        # Generate the same kind of data for all the duped classes

        X[cls * points_per_class:(cls + 1) * points_per_class] = X_cls
        y[cls * points_per_class:(cls + 1) * points_per_class] = cls

    # Add noise to the duplicate points
    if noise > 0:
        X += np.random.normal(0, noise, size=(n_samples, n_features))

    if n_classes - duped_classes > 0:
        X_cls, y_cls = generate_multiclass_unlearning_challenging_dataset(
            points_per_class * (n_classes - duped_classes), n_classes - duped_classes, n_features)

        # add n_classes - duped_classes to y_cls
        y_cls += duped_classes

        X[duped_classes * points_per_class:] = X_cls
        y[duped_classes * points_per_class:] = y_cls

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def generate_boundary_dataset(n_samples, n_features=2, boundary_proportion=0.1):
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    num_boundary_points = int(boundary_proportion * n_samples)
    num_remaining_points = n_samples - num_boundary_points

    # Decision boundary function (e.g., a sine wave)
    def decision_boundary(x):
        return np.sin(x)

    # First 10% points along the decision boundary
    for i in range(num_boundary_points):
        y[i] = 1 if np.random.rand() > 0.5 else 0
        x1 = np.random.uniform(-10, 10)
        x2 = decision_boundary(x1) + (-0.5 if y[i] == 0 else 0.5)
        X[i] = [x1, x2]

    for i in range(num_boundary_points, int(0.5 * n_samples)):
        x1 = np.random.uniform(-10, 10)
        if np.random.rand() > 0.5:
            x2 = decision_boundary(x1) + np.random.uniform(1.5, 3)  # Class 1 above the boundary
            y[i] = 1
        else:
            x2 = decision_boundary(x1) - np.random.uniform(1.5, 3)
            y[i] = 0
        X[i] = [x1, x2]


    # Remaining 90% points away from the decision boundary
    for i in range(int(0.5 * n_samples), n_samples):
        x1 = np.random.uniform(-10, 10)
        if np.random.rand() > 0.5:
            x2 = 1 + np.random.uniform(0, 1)  # Class 1 above the boundary
            y[i] = 1
        else:
            x2 = -1 - np.random.uniform(0, 1)  # Class 0 below the boundary
            y[i] = 0
        X[i] = [x1, x2]

    # X, y = shuffle(X, y)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return X, y


def synthetic_dataloaders(
        batch_size=128,
        data_dir="datasets/cifar10",
        num_workers=2,
        random_to_replace: int = None,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
        n_classes=2,
        n_features=2,
        n_samples=500,
):
    print("Dataset Information: Synthetic Dataset")

    seed = 2
    np.random.seed(seed)

    # X, y = sklearn.datasets.make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
    #                                             n_redundant=0, n_clusters_per_class=1, random_state=seed)

    # X, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
    # X = np.array([x.astype(float) for x in list(X)])
    # y = np.array([x.astype(int) for x in list(y)])

    # X, y = sklearn.datasets.make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_features, random_state=seed)

    alpha_1 = 1
    alpha_2 = 1
    beta_1 = 0.25
    beta_2 = -0.25

    num_points_per_line = 0.15 * n_samples

    X, y = generate_duplicate_dataset(n_samples, n_features, duplicate_proportion=0.1, noise=0)

    # X, y = generate_all_duplicate_classes_dataset(n_samples, n_features, n_classes, duped_classes=int(0.4 * n_classes),
    #                                               noise=0)

    # X, y = generate_boundary_dataset(n_samples, n_features, boundary_proportion=0.08)

    # X = np.zeros((n_samples, n_features))
    # y = np.zeros(n_samples)
    #
    # midpoint_0 = 0
    # midpoint_1 = 0
    #
    # # The first line is y = alpha_1 * x + beta_1
    # for i in range(int(num_points_per_line/2)):
    #     x = np.random.uniform(-10, 10)
    #     y_temp = alpha_1 * np.tanh(x + 1) + beta_1
    #     # y_temp = np.random.normal(-2, 1)
    #     X[i] = [x, y_temp]
    #     midpoint_0 += y_temp
    #     y[i] = 0
    #
    # midpoint_0 /= num_points_per_line/2
    #
    # # The second line is y = alpha_2 * x + beta_2
    # for i in range(int(num_points_per_line/2), int(num_points_per_line - 1)):
    #     x = np.random.uniform(-10, 10)
    #     y_temp = alpha_2 * np.tanh(x) + beta_2
    #     # y_temp = np.random.normal(2, 1)
    #     X[i] = [x, y_temp]
    #     midpoint_1 += y_temp
    #     y[i] = 1
    #
    # midpoint_1 /= num_points_per_line/2
    #
    # # The rest of the points are random noise added to the parameters and generated
    # for i in range(int(num_points_per_line), n_samples):
    #     x = np.random.uniform(-10, 10)
    #
    #     if np.random.uniform() < 0.5:
    #         y_temp = np.random.normal(1, 1)
    #         y_val = 0
    #     else:
    #         y_temp = np.random.normal(-1, 1)
    #         y_val = 1
    #
    #     X[i] = [x, y_temp]
    #     y[i] = y_val
    #
    # X = X.astype(np.float32)
    # y = y.astype(np.int64)

    X_train = X[:int(0.9 * n_samples)]
    y_train = y[:int(0.9 * n_samples)]
    X_test = X[int(0.9 * n_samples):]
    y_test = y[int(0.9 * n_samples):]

    train_set = SyntheticDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_set = SyntheticDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 0.1 * n_samples:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes_synthetic(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def cifar10_dataloaders(
        batch_size=128,
        data_dir="datasets/cifar10",
        num_workers=2,
        random_to_replace: int = None,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def mnist_dataloaders(
        batch_size=128,
        data_dir="datasets/mnist",
        num_workers=2,
        random_to_replace: int = None,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(3),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(3),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: MNIST\t 60000 images for training \t 10000 images for testing\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = MNIST(data_dir, train=True, transform=train_transform, download=True)

    test_set = MNIST(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == int(len(train_set.data) / 10):
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader

def usps_dataloaders(
        batch_size=128,
        data_dir="datasets/usps",
        num_workers=2,
        random_to_replace: int = None,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(3),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(3),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: USPS\t  7,291 images for training \t 2,007 images for testing\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = USPS(data_dir, train=True, transform=train_transform, download=True)

    test_set = USPS(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == int(len(train_set.data) / 10):
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader

def census_income_dataloaders(
        batch_size=128,
        data_dir="datasets/census_income",
        num_workers=2,
        random_to_replace: int = None,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: Census Income\t 60000 images for training \t 10000 images for testing\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = MNIST(data_dir, train=True, transform=train_transform, download=True)

    test_set = MNIST(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == int(len(train_set.data) / 10):
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader

def replace_indexes_synthetic(
        dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False
):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = indexes
        dataset.data[indexes] = dataset.data[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
            dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        try:
            dataset.targets[indexes] = -dataset.targets[indexes] - 1
        except:
            try:
                dataset.labels[indexes] = -dataset.labels[indexes] - 1
            except:
                dataset._labels[indexes] = -dataset._labels[indexes] - 1


def replace_indexes(
        dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False
):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes)
        )
        dataset.data[indexes] = dataset.data[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
            dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        try:
            dataset.targets[indexes] = -dataset.targets[indexes] - 1
        except:
            try:
                dataset.labels[indexes] = -dataset.labels[indexes] - 1
            except:
                try:
                    dataset._labels[indexes] = -dataset._labels[indexes] - 1
                except:
                    return


def replace_class(
        dataset: torch.utils.data.Dataset,
        class_to_replace: int,
        num_indexes_to_replace: int = None,
        seed: int = 0,
        only_mark: bool = False,
):
    if class_to_replace == -1:
        try:
            indexes = np.flatnonzero(np.ones_like(dataset.targets))
        except:
            try:
                indexes = np.flatnonzero(np.ones_like(dataset.labels))
            except:
                indexes = np.flatnonzero(np.ones_like(dataset._labels))
    else:
        try:
            indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
        except:
            try:
                indexes = np.flatnonzero(np.array(dataset.labels) == class_to_replace)
            except:
                indexes = np.flatnonzero(np.array(dataset._labels) == class_to_replace)

    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes
        ), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
        print(f"Replacing indexes {indexes}")
    replace_indexes(dataset, indexes, seed, only_mark)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = cifar10_dataloaders()
    for i, (img, label) in enumerate(train_loader):
        print(torch.unique(label).shape)
