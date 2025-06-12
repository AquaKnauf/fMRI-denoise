
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm  
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


from unet.dataset import FMRI3DDataset
from unet.model import UNet3DfMRI
#from unet.evaluate import evaluate_model


# -----------------------------
# Training & Evaluation
# -----------------------------

def fft_loss(pred, target):
    pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
    target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
    return torch.mean(torch.abs(pred_fft - target_fft))

def train_model(model: nn.Module, dataloader: DataLoader, device, epochs: int = 5, lr: float = 1e-4, writer: SummaryWriter = None, resume_from_epoch: int = 0):
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='none')  # We'll do masking manually

    fft_loss_weight = 0.05  # Can be tuned

    global_step = 0

    for epoch in range(resume_from_epoch, epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # Fix: match depth if model output is smaller
            if pred.shape[4] < y.shape[4]:
                y = y[..., :pred.shape[4]]
            elif pred.shape[4] > y.shape[4]:
                pred = pred[..., :y.shape[4]]

            # Per-voxel squared error
            per_voxel_loss = (pred - y) ** 2

            # Create brightness masks
            threshold = 0.3
            bright_mask = (y > threshold).float()
            dark_mask = 1.0 - bright_mask

            # Masked losses
            bright_loss = (per_voxel_loss * bright_mask).sum() / bright_mask.sum().clamp(min=1.0)
            dark_loss = (per_voxel_loss * dark_mask).sum() / dark_mask.sum().clamp(min=1.0)

            # Final weighted voxel loss
            loss = 0.1 * bright_loss + 0.9 * dark_loss
            loss *= 1e2  # Optional scaling

            # Add FFT frequency loss
            fft_loss_val = fft_loss(pred, y)
            loss += fft_loss_weight * fft_loss_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # Logging
            if writer and (global_step % 10 == 0):
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Loss/fft", fft_loss_val.item(), global_step)

                pred_np = pred.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()

                psnr_total, ssim_total = 0.0, 0.0

                for i in range(pred_np.shape[0]):
                    pred_vol = np.squeeze(pred_np[i])
                    y_vol = np.squeeze(y_np[i])

                    psnr = peak_signal_noise_ratio(y_vol, pred_vol, data_range=1.0)
                    ssim = structural_similarity(y_vol, pred_vol, data_range=1.0)
                    psnr_total += psnr
                    ssim_total += ssim

                writer.add_scalar("Train/PSNR", psnr_total / pred_np.shape[0], global_step)
                writer.add_scalar("Train/SSIM", ssim_total / pred_np.shape[0], global_step)

            if writer and (batch_idx % (len(dataloader) // 10 + 1) == 0):
                with torch.no_grad():
                    x_slice = x[0, 0, :, :, x.shape[4] // 2].cpu().unsqueeze(0)
                    y_slice = y[0, 0, :, :, y.shape[4] // 2].cpu().unsqueeze(0)
                    pred_slice = pred[0, 0, :, :, pred.shape[4] // 2].cpu().unsqueeze(0)

                    grid = make_grid(torch.stack([x_slice, y_slice, pred_slice]), nrow=3, normalize=False)
                    writer.add_image(f"Epoch_{epoch+1}/Input_GT_Pred", grid, global_step)

            global_step += 1

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        if writer:
            writer.add_scalar("Loss/epoch_avg", avg_loss, epoch + 1)

        torch.save(model.state_dict(), f"./runs/model_weights_epoch{epoch + 1}.pth")



if __name__ == "__main__":
    root_dir = "./fMRI/"
    batch_size = 4
    num_epochs = 2
    learning_rate = 1e-4
    writer = SummaryWriter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = FMRI3DDataset(root_dir)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = UNet3DfMRI().to(device)
    model.load_state_dict(torch.load("./runs/model_weights_epoch11.pth"))
    print("Starting training...")
    train_model(model, train_loader, device, epochs=12, lr=learning_rate, writer=writer, resume_from_epoch=11)

    print("Evaluating and visualizing...")
    writer.close()

