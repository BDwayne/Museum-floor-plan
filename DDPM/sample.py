# sample.py

import os
import torch
from torch.utils.data import DataLoader
from utils import BuildingDataset, save_images, custom_make_grid
from modules import UNet
from ddpm import Diffusion
import logging
from PIL import Image
import numpy as np
from utils import save_images

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")

def sample_and_save():
    # Set parameters directly in the code
    checkpoint_path = './autodl-tmp/SZYDDPM/models/SZYDDPM_Train/ckpt_epoch_1000.pt'   # Replace with your checkpoint path
    dataset_path = './autodl-tmp/SZYDDPM'            # Replace with your dataset path
    output_dir = './autodl-tmp/SZYDDPM/results/SZYDDPM_Train'                 # Directory to save generated images
    image_name='test_save.png'
    batch_size = 4                                   # Batch size for sampling
    image_size = 256                                 # Image size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to use

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device)

    # Load the model
    model = UNet(c_in=2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Initialize the diffusion process
    diffusion = Diffusion(img_size=image_size, device=device)

    # Prepare boundary images and structural info
    # For sampling, we can load a batch from the dataset
    dataset = BuildingDataset(
        boundary_dir=os.path.join(dataset_path, 'boundary/train'),
        labels_dir=os.path.join(dataset_path, 'labels/train'),
        connections_dir=os.path.join(dataset_path, 'connections/train'),
        image_size=image_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get a batch of data
    boundary_images, _, structural_info = next(iter(dataloader))
    boundary_images = boundary_images.to(device).unsqueeze(1).float()
    structural_info = structural_info.to(device)

    # Sample images using the model
    sampled_images = diffusion.sample(model, boundary_images, structural_info, n=boundary_images.size(0))
    # sampled_images shape: [batch_size, H, W, 3]

    # Save the RGB images
    save_images(sampled_images, output_dir,image_name)

    logging.info(f"Saved {sampled_images.shape[0]} color images to {output_dir}")

if __name__ == '__main__':
    sample_and_save()
