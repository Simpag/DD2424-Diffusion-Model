import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset


def cifar_10_transformed():
    data_transforms = [
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # Scales into [0, 1]
        shift_image  # Shift to [-1, 1]
    ]
    transform = T.Compose(data_transforms)

    # Load data locally or download it
    train_data = torchvision.datasets.CIFAR10(root="./datasets/", download=True, transform=transform, train=True)
    test_data = torchvision.datasets.CIFAR10(root="./datasets/", download=True, transform=transform, train=False)

    return train_data, test_data


def shift_image(t):
    return (t * 2) - 1


def load_data(train_data, test_data, batch_size, num_workers):
    dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return dataloader_train, dataloader_test

def plot_images(images, rows=None, cols=None, labels=None):
    if rows is None or cols is None:
        rows = 1
        cols = len(images)

    if type(images) == np.ndarray:
        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Display each image
        for i in range(4):
            axes[i].imshow(images[i])
            axes[i].axis('off')  # Hide the axes

        # If there are any remaining axes, hide them
        for i in range(4, rows * cols):
            axes[i].axis('off')
    else:
        #plt.figure(figsize=(16, 16))
        fig, axs = plt.subplots(rows, cols, figsize=(16, 2*rows))

        # Flatten the axes array for easy iteration
        axs = axs.flatten()

        for j in range(rows*cols):
            axs[j].imshow(images[j].permute(1, 2, 0).cpu())
            if labels is not None and j < cols:
                axs[j].set_title(labels[j]) # Set the title to the label
            axs[j].axis('off')

            #plt.imshow(torch.cat([
            #    torch.cat([i for i in images.cpu()], dim=-1),
            #], dim=-2).permute(1, 2, 0).cpu())

    plt.tight_layout()
    plt.show()
