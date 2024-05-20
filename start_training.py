import torch
import numpy as np

from DiffusionModel.diffusion_model import DiffusionModel
from DiffusionModel.trainer import Trainer
from utils import cifar_10_transformed, plot_images

if __name__ == "__main__":
    batch_size = 10 
    num_workers = 4
    lr = 3e-3
    epochs = 200
    train_data, test_data = cifar_10_transformed()
    use_amp = True 
    img_size = train_data.data[0].shape[0]
    cfg_strength = 3
    in_channels = train_data.data[0].shape[-1]
    out_channels = train_data.data[0].shape[-1] 
    encoder_decoder_layers = (64,128,256) 
    bottleneck_layers = (512,512)
    UNet_embedding_dimensions = 256 
    time_dimension = 256
    num_classes = len(train_data.classes)
    noise_steps = 1000
    beta_start = 1e-4
    beta_end = 2e-2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model = False
    validation = True

    model = DiffusionModel(in_channels, out_channels, encoder_decoder_layers, bottleneck_layers, UNet_embedding_dimensions, time_dimension, num_classes, noise_steps, beta_start, beta_end, device, compile_model)
    trainer = Trainer(model, batch_size, num_workers, lr, device, epochs, train_data, test_data, use_amp, img_size, cfg_strength, validation)
    trainer.fit()

    #### save model
    #model.save_model("test.pt", trainer.optimizer, trainer.scaler)