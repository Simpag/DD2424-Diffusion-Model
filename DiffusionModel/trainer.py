from diffusion_model import DiffusionModel
from utils import load_data, plot_images
import torch
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import wandb
import numpy as np


class Trainer:

    def __init__(self, model: DiffusionModel, batch_size: int, num_workers: int, lr: float, device: str, epochs: int,
                 train_data: object, test_data: object, use_amp: bool, img_size: int, cfg_strength: float, validation: bool):
        self.model = model
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
        self.cfg_strength = cfg_strength
        self.validation = validation

    def log_images(self):
        "Log images to wandb and save them to disk"
        labels = torch.arange(len(self.train_dataloader.classes)).long().to(self.device)
        # sample(self, image_size: int, image_channels: int, labels: list, cfg_strength: int)
        sampled_images = self.model.sample(self.img_size, 3, labels, self.cfg_strength)
        wandb.log(
            {"sampled_images": [wandb.Image(img.permute(1, 2, 0).squeeze().cpu().numpy()) for img in sampled_images]})


    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.model.noise_steps, size=(n,))


    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.model.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.model.alpha_hat[t])
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon


    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

    def train_epoch(self, epoch, train=True):
        avg_loss = 0.
        if train:
            self.model.train()
        else:
            self.model.eval()
        for i, (images, labels) in tqdm(self.train_dataloader):
            with torch.autocast("cuda", enabled=self.use_amp) and (
            torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.loss(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
                if i % 100 == 0:
                    wandb.log({"train_mse": loss.item(),
                               "learning_rate": self.scheduler.get_last_lr()[0]},
                              step=epoch * len(self.train_dataloader) + i)
        return avg_loss.mean().item()

    def fit(self):
        for epoch in tqdm(range(self.epochs)):
            self.train_epoch(epoch, train=True)

            #  validation
            if self.validation:
                avg_loss = self.train_epoch(epoch, train=False)
                wandb.log({"val_mse": avg_loss})

            #  log predictions
            if epoch % 100 == 0:
                self.log_images()

