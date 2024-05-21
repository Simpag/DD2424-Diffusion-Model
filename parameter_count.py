import copy
import torch

from DiffusionModel.diffusion_model import DiffusionModel

def count_parameters(model, only_trainable):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    img_size = 32
    in_channels = 3
    out_channels = 3
    encoder_decoder_layers = (64,128,256,512) 
    bottleneck_layers = (1024,)
    UNet_embedding_dimensions = 256 
    time_dimension = 256
    num_classes = 10
    noise_steps = 1000
    beta_start = 1e-4
    beta_end = 2e-2


    model = DiffusionModel(in_channels, out_channels, encoder_decoder_layers, bottleneck_layers, UNet_embedding_dimensions, time_dimension, num_classes, noise_steps, beta_start, beta_end, "cpu", compile_model=False)

    print("Trainable parameters: ", count_parameters(model, True))
    print("Including non-trainable parameters: ", count_parameters(model, False))