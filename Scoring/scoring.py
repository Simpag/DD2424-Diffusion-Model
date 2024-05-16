import typing
import torch
from torchmetrics.image.fid import FrechetInceptionDistance # https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
from torchmetrics.image.inception import InceptionScore # https://lightning.ai/docs/torchmetrics/stable/image/inception_score.html

def evaluate_generator(generated_images: torch.Tensor, real_images: torch.Tensor, normalized_images=False) -> typing.Tuple[float, float]:
    """
    Computes and returns the FID and IS scores. First float is mean fid value, second is mean inception score 
    over subsets and last float is the standard deviation of inception score over subsets.
    Set normalized_images to true if images are normalized to 0-1, else false if images are in the range [0-255].
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check which feature to use
    image_size = generated_images.shape[1:]   # shape without batch

    fid = FrechetInceptionDistance(input_img_size=image_size, normalize=normalized_images).set_dtype(torch.float64).to(device)
    fid.update(imgs=real_images, real=True)
    fid.update(imgs=generated_images, real=False)
    fid_score = fid.compute().item()

    inception = InceptionScore(normalize=normalized_images).set_dtype(torch.float64).to(device)
    inception.update(generated_images)
    inception_score = inception.compute().item()

    return (fid_score, *inception_score)