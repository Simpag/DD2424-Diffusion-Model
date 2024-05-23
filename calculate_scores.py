import torch
import platform
from tqdm import tqdm
import gc

from DiffusionModel.diffusion_model import DiffusionModel
from utils import cifar_10_transformed, plot_images, load_data
from Scoring.scoring import evaluate_generator

def log_results(res):
    with open("scores.txt", "a") as f:
        f.write(res)

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
    
    # Set model name to load (will also load ema_model_name)
    model_name = "ema_constantLR_continue.pt"

    # Set how many times we sample each class
    num_samples = 100 # There are 10_000 images in the test set
    batch_size = 10
    # sampled images will be num_samples*batch_size*num_classes

    #######################################
    labels = torch.arange(num_classes).repeat(batch_size).long().to(device)

    model = DiffusionModel(in_channels, out_channels, encoder_decoder_layers, bottleneck_layers, UNet_embedding_dimensions, time_dimension, num_classes, noise_steps, beta_start, beta_end, device, compile_model=False)
    model.load_model(model_name)
    model.compile_model()

    sampled_images = None
    for i in tqdm(range(num_samples), "Batch"):
        si = model.sample(img_size, out_channels, labels, cfg_strength)
        if sampled_images is None:
            sampled_images = si
        else:
            sampled_images = torch.cat([sampled_images, si], dim=0)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    real_images = torch.from_numpy(test_data.data).permute((0,3,1,2)).to(device)
    #assert real_images.shape[0] == sampled_images.shape[0], f"Sampled and real images must be same shape! Got sampled: {sampled_images.shape}, real: {real_images.shape}"

    fid_score, is_score, is_deviation = evaluate_generator(generated_images=sampled_images, real_images=real_images, num_labels=num_classes, normalized_images=False)
    print(f'FID: {fid_score}, IS: {is_score}, IS deviation: {is_deviation}')
    log_results(f'FID: {fid_score}, IS: {is_score}, IS deviation: {is_deviation}')
    