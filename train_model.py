import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from tqdm import tqdm

from unet.dataset import FMRI3DDataset
from unet.model import UNet3DfMRI

def fft_loss(pred, target):
    pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
    target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
    return torch.mean(torch.abs(pred_fft - target_fft))

def train_model(model: nn.Module, dataloader: DataLoader, device, epochs: int = 5, lr: float = 1e-4, writer: SummaryWriter = None, resume_from_epoch: int = 0):
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    fft_loss_weight = 0.05
    global_step = 0

    for epoch in range(resume_from_epoch, epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            
            
            pred = model(x)
            
            

            # Check for shape mismatches - this shouldn't happen with proper padding
            if pred.shape != y.shape:
                print(f"Warning: Shape mismatch! pred: {pred.shape}, target: {y.shape}")
                # Instead of cropping, investigate why shapes don't match
                min_shape = [min(pred.shape[i], y.shape[i]) for i in range(len(pred.shape))]
                pred = pred[:, :, :min_shape[2], :min_shape[3], :min_shape[4]]
                y = y[:, :, :min_shape[2], :min_shape[3], :min_shape[4]]

            # Standard MSE loss
            mse_loss = loss_fn(pred, y)
            
            # Add FFT frequency loss
            fft_loss_val = fft_loss(pred, y)
            loss = mse_loss + fft_loss_weight * fft_loss_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # Logging code remains the same...
            if writer and (global_step % 10 == 0):
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Loss/mse", mse_loss.item(), global_step)
                writer.add_scalar("Loss/fft", fft_loss_val.item(), global_step)

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

