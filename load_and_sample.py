import torch
import platform

from DiffusionModel.diffusion_model import DiffusionModel
from utils import cifar_10_transformed, plot_images

if __name__ == "__main__":
    if platform.system() == 'Linux':
        print("Setting high float32 matmul precision")
        torch.set_float32_matmul_precision('high')

    train_data, test_data = cifar_10_transformed()
    img_size = train_data.data[0].shape[0]
    cfg_strength = 3
    in_channels = train_data.data[0].shape[-1]
    out_channels = train_data.data[0].shape[-1] 
    encoder_decoder_layers = (64,128,256,512) 
    bottleneck_layers = (1024,)
    UNet_embedding_dimensions = 256 
    time_dimension = 256
    num_classes = len(train_data.classes)
    noise_steps = 1000
    beta_start = 1e-4
    beta_end = 2e-2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set model name to load, None if not loading
    model_name = None

    # Set how many times we sample each class
    num_samples = 3


    model = DiffusionModel(in_channels, out_channels, encoder_decoder_layers, bottleneck_layers, UNet_embedding_dimensions, time_dimension, num_classes, noise_steps, beta_start, beta_end, device, compile_model=False)

    #### save model
    if model_name is not None:
        model.load_model(model_name)

    labels = torch.arange(num_classes).repeat(num_samples).long().to(device)
    sampled_images = model.sample(img_size, out_channels, labels, cfg_strength)

    plot_images(sampled_images, num_samples, num_classes, train_data.classes)