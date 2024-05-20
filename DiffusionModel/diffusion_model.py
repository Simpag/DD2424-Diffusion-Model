import torch
import torch.nn as nn
import tqdm
import os
import wandb

from UNet.unet import UNet

class DiffusionModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, encoder_decoder_layers: list, bottleneck_layers: list, UNet_embedding_dimensions: int, time_dimension: int, num_classes: int, noise_steps: int, beta_start: float, beta_end: float, device: str, compile_model=True) -> None:
        """
        Diffusion Model module

        in_channels                 := Number of input channels
        out_channels                := Number of output channels
        encoder_decoder_layers      := Three values for the encoder/decoder
        bottleneck_layers           := Bottleneck dimension, at least one value
        UNet_embedding_dimensions   := Embedding dimensions for Up and Down modules
        time_dimension              := Time embedding dimension 
        num_classes                 := Number of classes in the dataset
        device                      := Which device the model is running on
        compile_model               := If it should compile the model or not

        Example usage:
        model = DiffusionModel(in_channels=3, out_channels=3, encoder_decoder_layers=(64,128,256), bottleneck_layers=(512,512), UNet_embedding_dimensions=256, time_dimension=256, num_classes=10, noise_steps=1000, beta_start=1e-4, beta_end=2e-2, device="cpu")
        """
        super().__init__()
        self.device = device
        self.noise_steps = noise_steps

        self.beta = self._generate_noise_schedule(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.model = UNet(in_channels, out_channels, encoder_decoder_layers, bottleneck_layers, UNet_embedding_dimensions, time_dimension, num_classes, device)
        self.model = self.model.to(device)
        if compile_model:
            print("Compiled model")
            self.model = torch.compile(self.model, mode="reduce-overhead")


    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
       return self.model(x, t, y)
    
    
    def sample(self, image_size: int, image_channels: int, labels: list, cfg_strength: int):
        """
        Sample images from diffusion model.

        image_size      := Image width (must equal height)
        image_channels  := How many image channels to generate
        labels          := List of labels to generate, will generate len(labels) images
        cfg_strength    := Classifier-free guidance strength, set to 3 in paper (linear interpolation between classifier free and classified generation)
        """
        # https://arxiv.org/pdf/2207.12598.pdf algorithm 2
        num_samples = len(labels)
        self.model.eval()
        with torch.inference_mode():
            z = torch.randn((num_samples, image_channels, image_size, image_size)).to(self.device) # initial random noise image
            for i in tqdm(reversed(range(1, self.noise_steps)), "Generating images"):
                t = (torch.ones(num_samples) * i).long().to(self.device) # time step i for every noise step
                conditional_predicted_noise = self.model(z, t, labels)
                
                if cfg_strength > 0: # dont need to compute unconditioned noise if cfg_strength is 0
                    unconditioned_predicted_noise = self.model(z, t, None)
                    predicted_noise = (1 + cfg_strength) * conditional_predicted_noise - cfg_strength * unconditioned_predicted_noise
                else:
                    predicted_noise = conditional_predicted_noise
                
                alpha = self.alpha[t]
                alpha_hat = self.alpha_hat[t]
                beta = self.beta[t]
                
                if i > 1:
                    noise = torch.randn_like(z) 
                else:
                    noise = 0 # no noise for last time-step

                sigma = (1 - alpha) / (torch.sqrt(1 - alpha_hat))       # TODO: check that this is correct
                x = (z - sigma * predicted_noise) / torch.sqrt(alpha)   # Line 4 in alg 2
                z = x + torch.sqrt(beta) * noise 

        imgs = (z.clamp(-1, 1) + 1) / 2         # Clamp values between -1 and 1, rescale to 0 to 1
        imgs = (imgs * 255).type(torch.uint8)   # Rescale pixel values to 0 and 255 
        return imgs


    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(os.path.join("models", file_name)))


    def save_model(self, file_name):
        "Save model locally and on wandb"
        if not os.path.exists("models/"):
            os.mkdir("models/")

        torch.save(self.model.state_dict(), os.path.join("models", file_name))
        at = wandb.Artifact("model", type="model", description=f"Model weights for Diffusion Model {file_name}")
        at.add_dir(os.path.join("models", file_name))
        wandb.log_artifact(at)


    def _generate_noise_schedule(self, start, end, steps):
        return torch.linspace(start, end, steps)
    
