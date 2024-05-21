import torch
import numpy as np
import wandb
import platform

from DiffusionModel.diffusion_model import DiffusionModel
from DiffusionModel.trainer import Trainer
from utils import cifar_10_transformed, plot_images

if __name__ == "__main__":
    if platform.system() == 'Linux':
        print("Setting high float32 matmul precision")
        torch.set_float32_matmul_precision('high')

    wandb.login()

    batch_size = 32 
    num_workers = 2
    lr = 5e-3
    ema_decay = 0.999
    epochs = 1
    train_data, test_data = cifar_10_transformed()
    use_amp = True 
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
    compile_model = True # Only available on linux, will be very slow at the start but will ramp up
    validation = True
    validation_logging_interval = 10
    image_logging_interval = 10
    model_name = "test.pt"

    print("Running on device: ", device)

    run = wandb.init(project="Diffusion Model", config={
            "dataset": "CIFAR10",
            "learning_rate": lr,
            "batch size": batch_size,
            "epochs": epochs,
            "cfg_strength": cfg_strength,
            "noise_steps": noise_steps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "encoder_decoder_layers": encoder_decoder_layers,
            "bottleneck_layers": bottleneck_layers,
            "UNet_embedding_dimensions": UNet_embedding_dimensions,
            "time_dimension": time_dimension,
        }
    )

    try:
        model = DiffusionModel(in_channels, out_channels, encoder_decoder_layers, bottleneck_layers, UNet_embedding_dimensions, time_dimension, num_classes, noise_steps, beta_start, beta_end, device, compile_model)
        trainer = Trainer(model, ema_decay, batch_size, num_workers, lr, device, epochs, train_data, test_data, use_amp, img_size, cfg_strength, validation)
        trainer.fit(validation_logging_interval, image_logging_interval)

        #### save models
        model.save_model(model_name, trainer.optimizer, trainer.scaler)
        trainer.ema_model.save_model("ema_" + model_name, trainer.optimizer, trainer.scaler)
    except KeyboardInterrupt:
        model.save_model(model_name, trainer.optimizer, trainer.scaler)
        trainer.ema_model.save_model("ema_" + model_name, trainer.optimizer, trainer.scaler)