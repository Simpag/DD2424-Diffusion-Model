import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset


def cifar_10_transformed():
    """
    Applies transformations to the CIFAR-10 dataset including random horizontal flip and normalization.

    Returns:
        train_data := Transformed training data
        test_data  := Transformed test data
    """
    # Define the sequence of transformations
    data_transforms = [
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # Scales image pixel values into [0, 1]
        shift_image  # Shift pixel values to [-1, 1]
    ]
    transform = T.Compose(data_transforms)

    # Load CIFAR-10 dataset either from file or download with transformations applied
    train_data = torchvision.datasets.CIFAR10(root="./datasets/", download=True, transform=transform, train=True)
    test_data = torchvision.datasets.CIFAR10(root="./datasets/", download=True, transform=transform, train=False)

    return train_data, test_data


def shift_image(t):
    """
    Shifts image pixel values from [0, 1] range to [-1, 1] range.

    Parameters:
        t := Tensor representing the image

    Returns:
        Tensor with pixel values shifted to [-1, 1]
    """
    return (t * 2) - 1


def load_data(train_data, test_data, batch_size, num_workers):
    """
    Loads training and test data into DataLoader objects.

    Parameters:
        train_data := Training data
        test_data  := Test data
        batch_size := Number of samples per batch
        num_workers:= Number of subprocesses to use for data loading

    Returns:
        dataloader_train := DataLoader for training data
        dataloader_test  := DataLoader for test data
    """
    dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return dataloader_train, dataloader_test

def plot_images(images, rows=None, cols=None, labels=None, save_file=None):
    """
    Plots a grid of images using Matplotlib.

    Parameters:
        images := List or array of images to plot
        rows   := Number of rows in the grid (optional)
        cols   := Number of columns in the grid (optional)
        labels := Labels for the images (optional)
    """
    if rows is None or cols is None:
        rows = 1
        cols = len(images)

    if type(images) == np.ndarray:
        # Create subplots for numpy array images
        fig, axes = plt.subplots(rows, cols)

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
        # Create subplots for torch tensors
        fig, axs = plt.subplots(rows, cols, figsize=(16, 16))

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

    plt.subplots_adjust(wspace=0, hspace=0)  # Adjust these values as needed

    if save_file is not None:
        plt.savefig(save_file)

    plt.show()
