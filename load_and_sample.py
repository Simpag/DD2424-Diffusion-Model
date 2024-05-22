import torch
import platform
import numpy as np

from DiffusionModel.diffusion_model import DiffusionModel
from utils import cifar_10_transformed, plot_images, load_data
from Scoring.scoring import evaluate_generator

if __name__ == "__main__":
    if platform.system() == 'Linux':
        print("Setting high float32 matmul precision")
        torch.set_float32_matmul_precision('high')

    train_data, test_data = cifar_10_transformed()
    train_dataloader, test_dataloader = load_data(train_data, test_data, 32, 2)
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
    model_name = "ema_test.pt"

    # Set how many times we sample each class
    num_samples = 1

    model = DiffusionModel(in_channels, out_channels, encoder_decoder_layers, bottleneck_layers, UNet_embedding_dimensions, time_dimension, num_classes, noise_steps, beta_start, beta_end, device, compile_model=False)

    #### save model
    if model_name is not None:
        model.load_model(model_name)

    labels = torch.arange(num_classes).repeat(num_samples).long().to(device)
    sampled_images = model.sample(img_size, out_channels, labels, cfg_strength)

    real_images_sample = np.random.choice(len(test_data.data), len(sampled_images))
    real_iamges_sample = test_data.data[real_images_sample, :, :, :]
    fid_score, is_score, is_deviation = evaluate_generator(generated_images=sampled_images, real_images=torch.from_numpy(real_iamges_sample).permute((0,3,1,2)).to(device), num_labels=num_classes, normalized_images=False)


    print(f'FID: {fid_score}, IS: {is_score}, IS deviation: {is_deviation}')
    plot_images(sampled_images, num_samples, num_classes, train_data.classes)