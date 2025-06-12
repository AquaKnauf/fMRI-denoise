import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
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
        for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            y_cropped = UNet3DfMRI.crop_to_match(y, pred)

            # Convert tensors to numpy
            y_np = y_cropped.squeeze().cpu().numpy()
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

    root_dir = "./eval_data"

    device = torch.device("cpu")

    dataset = FMRI3DDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = UNet3DfMRI()
    model.load_state_dict(torch.load("model_weights_.pth"))
    model.to(device)
    model.eval()  

    print("Evaluating and visualizing...")
    evaluate_model(model, dataloader, device)