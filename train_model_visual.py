import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from tqdm import tqdm

from unet.dataset import FMRI3DDataset
from unet.model import UNet3DfMRI

from unet.utils import analyze_frequency_content, create_frequency_visualizations

# Updated training function with frequency analysis
def train_model_with_frequency_analysis(model, dataloader, device, epochs=5, lr=1e-4, writer=None, resume_from_epoch=0):
    """
    Training function with integrated frequency analysis.
    """
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    fft_loss_weight = 0.05
    global_step = 0

    for epoch in range(resume_from_epoch, epochs):
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            # Handle shape mismatch if it still occurs
            if pred.shape != y.shape:
                min_shape = [min(pred.shape[i], y.shape[i]) for i in range(len(pred.shape))]
                pred = pred[:, :, :min_shape[2], :min_shape[3], :min_shape[4]]
                y = y[:, :, :min_shape[2], :min_shape[3], :min_shape[4]]
            
            # Standard loss
            mse_loss = loss_fn(pred, y)
            
            # FFT loss
            pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
            target_fft = torch.fft.fftn(y, dim=(-3, -2, -1))
            fft_loss_val = torch.mean(torch.abs(pred_fft - target_fft))
            
            loss = mse_loss + fft_loss_weight * fft_loss_val
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            # Frequency analysis every 20 steps
            if writer and (global_step % 20 == 0):
                with torch.no_grad():
                    analyze_frequency_content(y, pred, writer, global_step, "Train/")
                    
                    # Detailed visualizations every 100 steps
                    if global_step % 100 == 0:
                        create_frequency_visualizations(x, y, pred, writer, global_step, "Train/")
            
            # Regular logging
            if writer and (global_step % 10 == 0):
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Loss/mse", mse_loss.item(), global_step)
                writer.add_scalar("Loss/fft", fft_loss_val.item(), global_step)
            
            global_step += 1
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
        if writer:
            writer.add_scalar("Loss/epoch_avg", avg_loss, epoch + 1)
        
        # Save model
        torch.save(model.state_dict(), f"./runs/model_weights_epoch{epoch + 1}.pth")


if __name__ == "__main__":
    root_dir = "./data"
    batch_size = 4
    num_epochs = 6
    learning_rate = 1e-4
    resume_epoch = 0
    writer = SummaryWriter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full dataset
    full_dataset = FMRI3DDataset(root_dir)

    # Train/test split (optional, only useful if evaluation is separate)
    train_dataset = full_dataset

    # Dataloader for training set only
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True, prefetch_factor=8
    )

    # Initialize and optionally load pretrained weights
    model = UNet3DfMRI().to(device)

    if resume_epoch > 0:
        weight_path = f"./runs/model_weights_epoch{resume_epoch}.pth"
        model.load_state_dict(torch.load(weight_path))
        print(f"Resuming training from epoch {resume_epoch} using {weight_path}")
    else:
        print("Starting training from scratch...")

    # Train
    train_model_with_frequency_analysis(
        model,
        train_loader,
        device,
        epochs=num_epochs,
        lr=learning_rate,
        writer=writer,
        resume_from_epoch=resume_epoch
    )

    print("Training complete.")
    writer.close()

