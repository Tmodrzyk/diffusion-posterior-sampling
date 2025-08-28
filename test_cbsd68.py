from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import PosteriorSampling
from guided_diffusion.measurements import (
    GaussialBlurOperator,
    Levin1BlurOperator,
    Levin4BlurOperator,
    UniformBlurOperator,
    PoissonNoise,
)
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import cbsd68, get_dataloader
from util.img_utils import clear_color
from util.logger import get_logger
import random
import numpy as np
from deepinv.loss.metric import SSIM, PSNR, LPIPS

from pathlib import Path


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--diffusion_config", type=str)
    parser.add_argument("--task_config", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Reproducibility
    set_seed(0)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)

    # assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    # "learn_sigma must be the same for model and diffusion configuartion."

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operators and noise
    operators = {
        "gaussian": GaussialBlurOperator(kernel_size=31, intensity=1.6, device=device),
        "levin1": Levin1BlurOperator(device=device),
        "levin4": Levin4BlurOperator(device=device),
        "uniform": UniformBlurOperator(device=device),
    }

    noise_levels = {
        "20": PoissonNoise(rate=0.07843),
        "40": PoissonNoise(rate=0.15686),
        "60": PoissonNoise(rate=0.23529),
    }

    # Create metrics
    ssim = SSIM()
    psnr = PSNR()
    lpips = LPIPS(device=device)

    # Prepare dataloader
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = cbsd68(root="./data/cbsd68", transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Dictionary to store metrics
    metrics = {}

    # Loop over all combinations
    for noise_level, noiser in noise_levels.items():
        for op_name, operator in operators.items():
            # Create conditioning method
            cond_method = PosteriorSampling(operator=operator, noiser=noiser, scale=0.3)
            measurement_cond_fn = cond_method.conditioning

            # Load diffusion sampler
            sampler = create_sampler(**diffusion_config)
            sample_fn = partial(
                sampler.p_sample_loop,
                model=model,
                measurement_cond_fn=measurement_cond_fn,
            )

            # Create output directory
            out_path = os.path.join(
                args.save_dir, f"test_cbsd68/{op_name}_{noise_level}/"
            )
            os.makedirs(out_path, exist_ok=True)
            for img_dir in ["input", "recon", "progress", "label"]:
                os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

            # Initialize metric lists
            psnr_vals = []
            ssim_vals = []
            lpips_vals = []

            # Do Inference
            for i, ref_img in enumerate(loader):
                logger.info(f"Inference for {op_name}, noise {noise_level}, image {i}")
                fname = str(i).zfill(5) + ".png"
                ref_img = ref_img.to(device)

                # Forward measurement model
                y = operator.forward(ref_img)
                y_n = noiser(y)

                # Sampling
                x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
                sample = sample_fn(
                    x_start=x_start, measurement=y_n, record=False, save_root=out_path
                )

                # Compute metrics
                psnr_val = psnr(sample, ref_img)
                ssim_val = ssim(sample, ref_img)
                lpips_val = lpips(sample, ref_img)

                # Accumulate metrics
                psnr_vals.append(psnr_val.item())
                ssim_vals.append(ssim_val.item())
                lpips_vals.append(lpips_val.item())

                # Save images
                plt.imsave(os.path.join(out_path, "input", fname), clear_color(y_n))
                plt.imsave(os.path.join(out_path, "label", fname), clear_color(ref_img))
                plt.imsave(os.path.join(out_path, "recon", fname), clear_color(sample))

            # Store metrics for this combination
            key = f"{op_name}_{noise_level}"
            metrics[key] = {
                "psnr": (np.mean(psnr_vals), np.std(psnr_vals)),
                "ssim": (np.mean(ssim_vals), np.std(ssim_vals)),
                "lpips": (np.mean(lpips_vals), np.std(lpips_vals)),
            }

            # Log metrics
            logger.info(f"Results for {key}:")
            logger.info(
                f"PSNR: {metrics[key]['psnr'][0]:.4f} ± {metrics[key]['psnr'][1]:.4f}"
            )
            logger.info(
                f"SSIM: {metrics[key]['ssim'][0]:.4f} ± {metrics[key]['ssim'][1]:.4f}"
            )
            logger.info(
                f"LPIPS: {metrics[key]['lpips'][0]:.4f} ± {metrics[key]['lpips'][1]:.4f}"
            )

            # Write results to file
            with open(
                os.path.join(args.save_dir, "test_cbsd68/metrics_results.txt"), "a"
            ) as f:
                f.write(f"\nResults for noise level {noise_level}:\n")
                f.write("-" * 50 + "\n")
                f.write(f"\n{op_name} blur:\n")
                f.write(
                    f"PSNR: {metrics[key]['psnr'][0]:.4f} ± {metrics[key]['psnr'][1]:.4f}\n"
                )
                f.write(
                    f"SSIM: {metrics[key]['ssim'][0]:.4f} ± {metrics[key]['ssim'][1]:.4f}\n"
                )
                f.write(
                    f"LPIPS: {metrics[key]['lpips'][0]:.4f} ± {metrics[key]['lpips'][1]:.4f}\n"
                )


if __name__ == "__main__":
    main()
