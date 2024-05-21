import torch
import torch.nn as nn
from tqdm import tqdm
import os
import wandb

from UNet.unet import UNet

class DiffusionModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, encoder_decoder_layers: list, bottleneck_layers: list, UNet_embedding_dimensions: int, time_dimension: int, num_classes: int, noise_steps: int, beta_start: float, beta_end: float, device: str, compile_model=True) -> None:
        """
         Initialize the Diffusion Model module.

         Args:
             in_channels (int): Number of input channels.
             out_channels (int): Number of output channels.
             encoder_decoder_layers (list): Three values for the encoder/decoder.
             bottleneck_layers (list): Bottleneck dimension, at least one value.
             UNet_embedding_dimensions (int): Embedding dimensions for Up and Down modules.
             time_dimension (int): Time embedding dimension.
             num_classes (int): Number of classes in the dataset.
             noise_steps (int): Number of noise steps.
             beta_start (float): Starting value for beta in the noise schedule.
             beta_end (float): Ending value for beta in the noise schedule.
             device (str): Device on which the model runs.
             compile_model (bool): If True, compile the model.

        Example usage:
        model = DiffusionModel(in_channels=3, out_channels=3, encoder_decoder_layers=(64,128,256), bottleneck_layers=(512,512), UNet_embedding_dimensions=256, time_dimension=256, num_classes=10, noise_steps=1000, beta_start=1e-4, beta_end=2e-2, device="cpu")
        """
        super().__init__()
        self.device = device
        self.noise_steps = noise_steps

        # Generate noise schedule and move it to the device
        self.beta = self._generate_noise_schedule(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Initialize the UNet model and move it to the device (CUDA/CPU)
        self.model = UNet(in_channels, out_channels, encoder_decoder_layers, bottleneck_layers, UNet_embedding_dimensions, time_dimension, num_classes, device)
        self.model = self.model.to(device)
        if compile_model:
            self.compile_model()


    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        # Forward pass through the UNet model
        return self.model(x, t, y)
    

    def sample(self, image_size: int, image_channels: int, labels: list, cfg_strength: float):
        """
        Sample images from diffusion model.

        image_size      := Image width (must equal height)
        image_channels  := How many image channels to generate
        labels          := List of labels to generate, will generate len(labels) images
        cfg_strength    := Classifier-free guidance strength, set to 3 in paper (linear interpolation between classifier free and classified generation)
        """
        # https://arxiv.org/pdf/2207.12598.pdf algorithm 2
        # Number of samples to generate
        num_samples = len(labels)
        self.model.eval()
        with torch.inference_mode():
            # Initial random noise image
            z = torch.randn((num_samples, image_channels, image_size, image_size)).to(self.device) # initial random noise image
            for i in tqdm(reversed(range(1, self.noise_steps)), f"Generating images ({num_samples})", total=self.noise_steps, position=2):
                # Time step i for every noise step
                t = (torch.ones(num_samples) * i).long().to(self.device) # time step i for every noise step
                conditional_predicted_noise = self.model(z, t, labels)

                # Compute unconditioned noise if cfg_strength > 0
                if cfg_strength > 0: # dont need to compute unconditioned noise if cfg_strength is 0
                    unconditioned_predicted_noise = self.model(z, t, None)
                    predicted_noise = (1 + cfg_strength) * conditional_predicted_noise - cfg_strength * unconditioned_predicted_noise
                else:
                    predicted_noise = conditional_predicted_noise

                # Calculate alpha, alpha_hat, and beta for current timestep
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # Add noise for all steps except the last one
                if i > 1:
                    noise = torch.randn_like(z) 
                else:
                    noise = 0 # no noise for last time-step

                # Update z using the diffusion equation
                sigma = (1 - alpha) / (torch.sqrt(1 - alpha_hat))       # TODO: check that this is correct
                x = (z - sigma * predicted_noise) / torch.sqrt(alpha)   # Line 4 in alg 2
                z = x + torch.sqrt(beta) * noise

        # Reverse transformation with correct RGB channel values
        imgs = (z.clamp(-1, 1) + 1) / 2         # Clamp values between -1 and 1, rescale to 0 to 1
        imgs = (imgs * 255).type(torch.uint8)   # Rescale pixel values to 0 and 255 
        return imgs


    def compile_model(self):
        # Compile the model to reduce overhead
        print("Compiled model")
        self.model = torch.compile(self.model, mode="reduce-overhead")


    def load_model(self, file_name, compile_model=False):
        """
        Load model from a file.
        Returns the state dict for optimizer and scaler
        Load them by:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
        """
        file_path = os.path.join("./models", file_name)
        checkpoint = torch.load(file_path, map_location = lambda storage, loc: storage.cuda(self.device))
        self.model.load_state_dict(checkpoint["model"])

        print("Loaded model!")

        if compile_model:
            self.compile_model()

        return checkpoint["optimizer"], checkpoint["scaler"]


    def save_model(self, file_name, optimizer, scaler):
        """
        Save model locally and on wandb.

        Args:
            file_name (str): Name of the file to save the model to.
            optimizer: Optimizer state.
            scaler: Scaler state.
        """
        if not os.path.exists("./models/"):
            os.mkdir("./models/")

        checkpoint = {"model": getattr(self.model, '_orig_mod', self.model).state_dict(), #self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict()}

        file_path = os.path.join("./models", file_name)
        torch.save(checkpoint, file_path)
        at = wandb.Artifact(name="model", type="model", description=f"Model weights for Diffusion Model {file_name}")
        at.add_file(local_path=file_path, name=file_name)
        wandb.log_artifact(at)


    def _generate_noise_schedule(self, start, end, steps):
        # Generate a linear noise schedule
        return torch.linspace(start, end, steps)
    
