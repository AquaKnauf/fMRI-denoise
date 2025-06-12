from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  
from torch.utils.tensorboard import SummaryWriter



def show_prediction(x : Tensor, y_true: Tensor, y_pred: Tensor, iteration: int) -> None:
    """
    Visualize middle slices of input, ground truth, and prediction.
    """
    x = x.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # Get middle slice from z-dimension
    slice_idx = x.shape[4] // 2

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    axs[0].imshow(x[0, 0, :, :, slice_idx], cmap='gray')
    axs[0].set_title("Input (Noisy)")
    axs[0].axis('off')

    axs[1].imshow(y_true[0, 0, :, :, slice_idx], cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(y_pred[0, 0, :, :, slice_idx], cmap='gray')
    axs[2].set_title("Prediction")
    axs[2].axis('off')

    plt.suptitle(f"Iteration {iteration}")
    plt.tight_layout()

    # Display the plot and clear it after a short pause
    plt.draw()  # Update the figure
    plt.show()


def log_images(writer: SummaryWriter, x: Tensor, y_true: Tensor, y_pred: Tensor, step: int) -> None:
    """
    Log matplotlib images to TensorBoard.
    """
    x = x.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    slice_idx = x.shape[4] // 2

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(x[0, 0, :, :, slice_idx], cmap='gray')
    axs[0].set_title("Input (Noisy)")
    axs[0].axis('off')

    axs[1].imshow(y_true[0, 0, :, :, slice_idx], cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(y_pred[0, 0, :, :, slice_idx], cmap='gray')
    axs[2].set_title("Prediction")
    axs[2].axis('off')

    plt.tight_layout()
    writer.add_figure("Prediction_vs_Truth", fig, global_step=step)
    plt.close(fig)


def show_slice(volume_tensor: Tensor, title="") -> None:
    """Show a middle axial slice of a 3D volume"""
    volume_np = volume_tensor.squeeze().cpu().numpy()
    z = volume_np.shape[2] // 2
    plt.imshow(volume_np[:, :, z], cmap='gray')
    plt.title(title)
    plt.axis('off')