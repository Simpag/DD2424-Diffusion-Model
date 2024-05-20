import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset


def get_cifar_10():
    data_transforms = [
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # Scales into [0, 1]
        T.Lambda(lambda t: (t * 2) - 1)  # Shift to [-1, 1]
    ]
    transform = T.Compose(data_transforms)

    # Load data locally or download it
    train_data = torchvision.datasets.CIFAR10(root="./datasets/", download=True, transform=transform, train=True)
    test_data = torchvision.datasets.CIFAR10(root="./datasets/", download=True, transform=transform, train=False)

    return ConcatDataset([train_data, test_data])


def load_data(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
