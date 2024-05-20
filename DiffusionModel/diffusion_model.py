import torch
import torch.nn as nn
import wandb
import os

class DiffusionModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, encoder_decoder_layers: list, bottleneck_layers: list, UNet_embedding_dimensions: int, time_dimension: int, num_classes: int, device: str) -> None:
        """"""
        pass
    
    def sample(self):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass