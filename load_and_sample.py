import torch

from DiffusionModel.diffusion_model import DiffusionModel
from utils import cifar_10_transformed, plot_images

if __name__ == "__main__":
    batch_size = 32 
    num_workers = 3
    lr = 3e-3
    epochs = 1
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
    validation_logging_interval = 5
    image_logging_interval = 100

    model = DiffusionModel(in_channels, out_channels, encoder_decoder_layers, bottleneck_layers, UNet_embedding_dimensions, time_dimension, num_classes, noise_steps, beta_start, beta_end, device, compile_model)

    #### save model
    model.load_model("test.pt")

    labels = torch.arange(num_classes).long().to(device)
    sampled_images = model.sample(img_size, out_channels, labels, cfg_strength)

    plot_images(sampled_images)