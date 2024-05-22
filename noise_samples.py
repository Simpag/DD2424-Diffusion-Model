import torch
import platform
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from DiffusionModel.diffusion_model import DiffusionModel
from utils import cifar_10_transformed, load_data

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
    model_name = "ema_constantLR.pt"

    # Set how many times we sample each class
    num_samples = 1

    # Set when to sample (reversed order, 1 is last step)
    #noise_samples = [1, 50, 100, 200, 400, 600, 800, 999]
    noise_samples = np.round(np.logspace(0,3,10))-1

    model = DiffusionModel(in_channels, out_channels, encoder_decoder_layers, bottleneck_layers, UNet_embedding_dimensions, time_dimension, num_classes, noise_steps, beta_start, beta_end, device, compile_model=False)

    #### save model
    if model_name is not None:
        model.load_model(model_name)

    labels = torch.arange(num_classes).repeat(num_samples).long().to(device)

    sampled_images = None

    num_samples = len(labels)
    model.eval()
    with torch.inference_mode():
        # Initial random noise image
        z = torch.randn((num_samples, out_channels, img_size, img_size)).to(device) # initial random noise image
        sampled_images = ((z.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        for i in tqdm(reversed(range(1, noise_steps)), f"Generating images ({num_samples})", total=noise_steps):
            # Time step i for every noise step
            t = (torch.ones(num_samples) * i).long().to(device) # time step i for every noise step
            conditional_predicted_noise = model(z, t, labels)

            # Compute unconditioned noise if cfg_strength > 0
            if cfg_strength > 0: # dont need to compute unconditioned noise if cfg_strength is 0
                unconditioned_predicted_noise = model(z, t, None)
                predicted_noise = (1 + cfg_strength) * conditional_predicted_noise - cfg_strength * unconditioned_predicted_noise
            else:
                predicted_noise = conditional_predicted_noise

            # Calculate alpha, alpha_hat, and beta for current timestep
            alpha = model.alpha[t][:, None, None, None]
            alpha_hat = model.alpha_hat[t][:, None, None, None]
            beta = model.beta[t][:, None, None, None]

            # Add noise for all steps except the last one
            if i > 1:
                noise = torch.randn_like(z) 
            else:
                noise = 0 # no noise for last time-step

            # Update z using the diffusion equation
            sigma = (1 - alpha) / (torch.sqrt(1 - alpha_hat))       # TODO: check that this is correct
            x = (z - sigma * predicted_noise) / torch.sqrt(alpha)   # Line 4 in alg 2
            z = x + torch.sqrt(beta) * noise

            if i in noise_samples:
                # Reverse transformation with correct RGB channel values
                imgs = (z.clamp(-1, 1) + 1) / 2         # Clamp values between -1 and 1, rescale to 0 to 1
                imgs = (imgs * 255).type(torch.uint8)   # Rescale pixel values to 0 and 255 

                sampled_images = torch.cat([sampled_images, imgs], dim=0)


    fig, axs = plt.subplots(num_classes, len(noise_samples), figsize=(10, 10))
    i = 0
    for col in range(len(noise_samples)):
        for row in range(num_classes):
            if col == 0:
                axs[row][col].text(-8, 16, test_data.classes[row], va='center', ha='right', rotation=0)

            axs[row][col].imshow(sampled_images[i].permute(1, 2, 0).cpu())
            i += 1

            if row == 0:
                axs[row][col].set_title(int(noise_samples[col])) # Set the title to the label

            axs[row][col].axis('off')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)  # Adjust these values as needed
    plt.show()