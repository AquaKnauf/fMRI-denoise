import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


from unet.dataset import FMRI3DDataset
from unet.model import UNet3DfMRI


def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)

    mse_list = []
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for i, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # Convert tensors to numpy (assuming shape: [1, 1, D, H, W])
            y_np = y.squeeze().cpu().numpy()
            pred_np = pred.squeeze().cpu().numpy()

            # Metrics per 3D volume
            mse_score = np.mean((y_np - pred_np) ** 2)
            psnr_score = psnr(y_np, pred_np, data_range=1.0)
            ssim_score = ssim(y_np, pred_np, data_range=1.0)

            mse_list.append(mse_score)
            psnr_list.append(psnr_score)
            ssim_list.append(ssim_score)



    print(f"Evaluation Results:")
    print(f"  MSE  : {np.mean(mse_list):.6f}")
    print(f"  PSNR : {np.mean(psnr_list):.2f} dB")
    print(f"  SSIM : {np.mean(ssim_list):.4f}")


if __name__ == "__main__":
    import os

    root_dir = "./validation"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FMRI3DDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet3DfMRI()
    model.load_state_dict(torch.load("runs/model_weights_epoch6.pth", map_location=device))

    print("Evaluating model...")
    evaluate_model(model, dataloader, device)
