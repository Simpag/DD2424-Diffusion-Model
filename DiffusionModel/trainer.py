from DiffusionModel.diffusion_model import DiffusionModel
from utils import load_data, plot_images
import torch
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import wandb
import numpy as np


class Trainer:

    def __init__(self, model: DiffusionModel, ema_model: DiffusionModel, ema_decay: float, batch_size: int, num_workers: int, lr: float, device: str, epochs: int,
                 train_data: object, test_data: object, use_amp: bool, img_size: int, cfg_strength: float,
                 validation: bool):
        """
        Initialize the Trainer class with the given parameters.

        Parameters:
        - model: Instance of the DiffusionModel.
        - ema_model: Copy of the DiffusionModel to store EMA, set to None if not using EMA
        - ema_decay: EMA coefficient
        - batch_size: Size of the batch for training.
        - num_workers: Number of workers for data loading.
        - lr: Learning rate for the optimizer.
        - device: Device to run the model on (e.g., 'cuda' or 'cpu').
        - epochs: Number of training epochs.
        - train_data: Training dataset.
        - test_data: Testing dataset.
        - use_amp: Boolean to use Automatic Mixed Precision (AMP) or not.
        - img_size: Size of the input images.
        - cfg_strength: Configuration strength for sampling.
        - validation: Boolean to indicate if validation should be performed.
        """
        self.model = model
        self.ema_model = ema_model
        self.ema_decay = ema_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_dataloader, self.test_dataloader = load_data(train_data, test_data, batch_size, num_workers)
        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = device
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr,
                                                       steps_per_epoch=len(self.train_dataloader), epochs=self.epochs)
        self.img_size = img_size
        self.img_channels = train_data.data[0].shape[-1]
        self.cfg_strength = cfg_strength
        self.validation = validation
        self.num_classes = len(train_data.classes)


    def log_images(self):
        """
        Log sampled images to Weights and Biases (wandb)
        """
        labels = torch.arange(self.num_classes).long().to(self.device)
        # Sample images from the model
        sampled_images = self.model.sample(self.img_size, self.img_channels, labels, self.cfg_strength)
        
        if self.ema_model is None:
            wandb.log(
                {
                    "sampled_images": [wandb.Image(img.permute(1, 2, 0).squeeze().cpu().numpy()) for img in sampled_images]
                }
            )
            return

        # Sample images from the model
        sampled_images_ema = self.ema_model.sample(self.img_size, self.img_channels, labels, self.cfg_strength)
        # Log images to wandb
        wandb.log(
                {
                    "sampled_images": [wandb.Image(img.permute(1, 2, 0).squeeze().cpu().numpy()) for img in sampled_images],
                    "sampled_images_ema": [wandb.Image(img.permute(1, 2, 0).squeeze().cpu().numpy()) for img in sampled_images_ema],
                }
            )
        

    def get_random_timesteps(self, n):
        """
        Generate random timesteps for noise addition.

        Parameters:
        - n: Number of random timesteps to generate.

        Returns:
        - Random timesteps tensor.
        """
        return torch.randint(low=1, high=self.model.noise_steps, size=(n,))


    def add_noise_to_images(self, x, t):
        """
        Adds noise to images for a certain timestep in the noise scheduler.

        Parameters:
        - x: Input images.
        - t: Timesteps for noise addition.

        Returns:
        - Noisy images and the added noise.
        """
        sqrt_alpha_hat = torch.sqrt(self.model.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.model.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def EMA(self):
        """
        Store Exponential Moving Average (EMA) of the weights during training
        """
        # https://arxiv.org/abs/2312.02696
        for current_model_param, ema_model_param in zip(self.model.parameters(), self.ema_model.parameters()):
            new = current_model_param.data
            old = ema_model_param.data
            ema_model_param.data = old * self.ema_decay + (1 - self.ema_decay) * new

    def train_one_step(self, loss):
        """
        Perform a single training step.

        Parameters:
        - loss: Computed loss for backpropagation.
        """
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # EMA update
        if self.ema_model is not None:
            self.EMA()

    def run_epoch(self, epoch, train=True):
        """
        Run a single epoch of training or validation.

        Parameters:
        - epoch: Current epoch number.
        - train: Boolean indicating whether to train or validate.

        Returns:
        - Average loss for the epoch.
        """
        avg_loss = 0.
        dataloader = None
        if train:
            self.model.train()
            dataloader = self.train_dataloader
        else:
            self.model.eval()
            dataloader = self.test_dataloader

        for i, (images, labels) in enumerate(tqdm(dataloader, "Training" if train else "Validation", position=1)):
            with torch.autocast("cuda", enabled=self.use_amp) and (
                    torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.get_random_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.add_noise_to_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.loss(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_one_step(loss)
                if i % 100 == 0:
                    wandb.log({"train_mse": loss.item(),
                               "learning_rate": self.scheduler.get_last_lr()[0]},
                              step=epoch * len(dataloader) + i)

        return avg_loss.mean().item()

    def fit(self, validation_logging_interval=5, image_logging_interval=100):
        """
        Fit the model to the training data.

        Parameters:
        - validation_logging_interval: Interval for logging validation metrics.
        - image_logging_interval: Interval for logging sampled images.
        """
        for epoch in tqdm(range(self.epochs), "Epoch", position=0):
            self.run_epoch(epoch, train=True)

            # Perform validation at specified intervals
            if self.validation and epoch % validation_logging_interval == 0:
                avg_loss = self.run_epoch(epoch, train=False)
                wandb.log({"val_mse": avg_loss})

            # Log sampled images at specified intervals
            if epoch % image_logging_interval == 0:
                self.log_images()
