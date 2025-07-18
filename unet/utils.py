import torch
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_frequency_spectrum(volume):
    """
    Compute 3D frequency spectrum of a volume.
    
    Args:
        volume: torch.Tensor of shape (B, C, D, H, W) or (D, H, W)
    
    Returns:
        magnitude_spectrum: Magnitude of the 3D FFT
        phase_spectrum: Phase of the 3D FFT
        power_spectrum: Power spectrum (magnitude squared)
    """
    if volume.dim() == 5:  # (B, C, D, H, W)
        volume = volume.squeeze()  # Remove batch and channel dims if present
    elif volume.dim() == 4:  # (C, D, H, W)
        volume = volume.squeeze(0)  # Remove channel dim
    
    # Compute 3D FFT
    fft_volume = torch.fft.fftn(volume, dim=(-3, -2, -1))
    
    # Get magnitude and phase
    magnitude_spectrum = torch.abs(fft_volume)
    phase_spectrum = torch.angle(fft_volume)
    power_spectrum = magnitude_spectrum ** 2
    
    return magnitude_spectrum, phase_spectrum, power_spectrum


def compute_radial_profile(spectrum):
    """
    Compute radially averaged power spectrum.
    
    Args:
        spectrum: 3D power spectrum tensor
    
    Returns:
        frequencies: Array of spatial frequencies
        radial_profile: Radially averaged power
    """
    spectrum_np = spectrum.cpu().numpy()
    d, h, w = spectrum_np.shape
    
    # Create coordinate grids
    z_coords = np.arange(d) - d // 2
    y_coords = np.arange(h) - h // 2
    x_coords = np.arange(w) - w // 2
    
    Z, Y, X = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    
    # Compute radial distance
    R = np.sqrt(Z**2 + Y**2 + X**2)
    
    # Shift zero frequency to center
    spectrum_shifted = np.fft.fftshift(spectrum_np)
    
    # Compute radial profile
    max_radius = int(np.min([d, h, w]) // 2)
    radial_profile = np.zeros(max_radius)
    frequencies = np.arange(max_radius)
    
    for r in range(max_radius):
        mask = (R >= r) & (R < r + 1)
        if np.sum(mask) > 0:
            radial_profile[r] = np.mean(spectrum_shifted[mask])
    
    return frequencies, radial_profile
    
def analyze_frequency_content(ground_truth, prediction, writer, global_step, prefix=""):
    """
    Comprehensive frequency analysis between ground truth and prediction.
    
    Args:
        ground_truth: torch.Tensor of shape (B, C, D, H, W)
        prediction: torch.Tensor of shape (B, C, D, H, W)
        writer: TensorBoard SummaryWriter
        global_step: Current training step
        prefix: Prefix for TensorBoard tags
    """
    batch_size = ground_truth.shape[0]
    
    # Process each sample in the batch
    for batch_idx in range(min(batch_size, 2)):  # Limit to first 2 samples to avoid overwhelming TB
        gt_sample = ground_truth[batch_idx]
        pred_sample = prediction[batch_idx]
        
        # Compute frequency spectra
        gt_mag, gt_phase, gt_power = compute_frequency_spectrum(gt_sample)
        pred_mag, pred_phase, pred_power = compute_frequency_spectrum(pred_sample)
        
        # 1. Log magnitude spectra statistics
        writer.add_scalar(f"{prefix}Frequency/GT_Magnitude_Mean_Sample_{batch_idx}", 
                         torch.mean(gt_mag).item(), global_step)
        writer.add_scalar(f"{prefix}Frequency/Pred_Magnitude_Mean_Sample_{batch_idx}", 
                         torch.mean(pred_mag).item(), global_step)
        
        writer.add_scalar(f"{prefix}Frequency/GT_Magnitude_Std_Sample_{batch_idx}", 
                         torch.std(gt_mag).item(), global_step)
        writer.add_scalar(f"{prefix}Frequency/Pred_Magnitude_Std_Sample_{batch_idx}", 
                         torch.std(pred_mag).item(), global_step)
        
        # 2. Log power spectrum statistics
        writer.add_scalar(f"{prefix}Frequency/GT_Power_Mean_Sample_{batch_idx}", 
                         torch.mean(gt_power).item(), global_step)
        writer.add_scalar(f"{prefix}Frequency/Pred_Power_Mean_Sample_{batch_idx}", 
                         torch.mean(pred_power).item(), global_step)
        
        # 3. Frequency domain loss metrics
        magnitude_diff = torch.mean(torch.abs(gt_mag - pred_mag))
        phase_diff = torch.mean(torch.abs(gt_phase - pred_phase))
        power_diff = torch.mean(torch.abs(gt_power - pred_power))
        
        writer.add_scalar(f"{prefix}Frequency/Magnitude_L1_Loss_Sample_{batch_idx}", 
                         magnitude_diff.item(), global_step)
        writer.add_scalar(f"{prefix}Frequency/Phase_L1_Loss_Sample_{batch_idx}", 
                         phase_diff.item(), global_step)
        writer.add_scalar(f"{prefix}Frequency/Power_L1_Loss_Sample_{batch_idx}", 
                         power_diff.item(), global_step)
        
        # 4. Compute and log radial profiles
        gt_freqs, gt_radial = compute_radial_profile(gt_power)
        pred_freqs, pred_radial = compute_radial_profile(pred_power)
        
        # Create radial profile comparison plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.semilogy(gt_freqs, gt_radial, 'b-', label='Ground Truth', linewidth=2)
        ax.semilogy(pred_freqs, pred_radial, 'r--', label='Prediction', linewidth=2)
        ax.set_xlabel('Spatial Frequency')
        ax.set_ylabel('Power (log scale)')
        ax.set_title(f'Radial Power Spectrum - Sample {batch_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        writer.add_figure(f"{prefix}Frequency/Radial_Power_Spectrum_Sample_{batch_idx}", 
                         fig, global_step)
        plt.close(fig)
        
        # 5. Log frequency band energies
        # Low frequencies (0-25% of max freq)
        # Mid frequencies (25-75% of max freq)  
        # High frequencies (75-100% of max freq)
        max_freq = len(gt_freqs)
        low_end = max_freq // 4
        high_start = 3 * max_freq // 4
        
        gt_low_energy = np.sum(gt_radial[:low_end])
        gt_mid_energy = np.sum(gt_radial[low_end:high_start])
        gt_high_energy = np.sum(gt_radial[high_start:])
        
        pred_low_energy = np.sum(pred_radial[:low_end])
        pred_mid_energy = np.sum(pred_radial[low_end:high_start])
        pred_high_energy = np.sum(pred_radial[high_start:])
        
        writer.add_scalar(f"{prefix}Frequency/GT_Low_Freq_Energy_Sample_{batch_idx}", 
                         gt_low_energy, global_step)
        writer.add_scalar(f"{prefix}Frequency/GT_Mid_Freq_Energy_Sample_{batch_idx}", 
                         gt_mid_energy, global_step)
        writer.add_scalar(f"{prefix}Frequency/GT_High_Freq_Energy_Sample_{batch_idx}", 
                         gt_high_energy, global_step)
        
        writer.add_scalar(f"{prefix}Frequency/Pred_Low_Freq_Energy_Sample_{batch_idx}", 
                         pred_low_energy, global_step)
        writer.add_scalar(f"{prefix}Frequency/Pred_Mid_Freq_Energy_Sample_{batch_idx}", 
                         pred_mid_energy, global_step)
        writer.add_scalar(f"{prefix}Frequency/Pred_High_Freq_Energy_Sample_{batch_idx}", 
                         pred_high_energy, global_step)
        
        # 6. Frequency band error ratios
        if gt_low_energy > 0:
            low_freq_error = abs(pred_low_energy - gt_low_energy) / gt_low_energy
            writer.add_scalar(f"{prefix}Frequency/Low_Freq_Relative_Error_Sample_{batch_idx}", 
                             low_freq_error, global_step)
        
        if gt_mid_energy > 0:
            mid_freq_error = abs(pred_mid_energy - gt_mid_energy) / gt_mid_energy
            writer.add_scalar(f"{prefix}Frequency/Mid_Freq_Relative_Error_Sample_{batch_idx}", 
                             mid_freq_error, global_step)
        
        if gt_high_energy > 0:
            high_freq_error = abs(pred_high_energy - gt_high_energy) / gt_high_energy
            writer.add_scalar(f"{prefix}Frequency/High_Freq_Relative_Error_Sample_{batch_idx}", 
                             high_freq_error, global_step)

    
    
def create_frequency_visualizations(noisy_input, ground_truth, prediction, writer, global_step, prefix=""):
    """
    Visualize spatial and frequency domain slices of noisy input, ground truth, and prediction.
    
    Args:
        noisy_input: torch.Tensor of shape (B, C, D, H, W)
        ground_truth: torch.Tensor of shape (B, C, D, H, W)
        prediction: torch.Tensor of shape (B, C, D, H, W)
        writer: TensorBoard SummaryWriter
        global_step: Current training step
        prefix: Prefix for TensorBoard tags
    """
    # Extract central slice in width axis (W)
    slice_idx = ground_truth.shape[4] // 2
    noisy_slice = noisy_input[0, 0, :, :, slice_idx].cpu().numpy()
    gt_slice = ground_truth[0, 0, :, :, slice_idx].cpu().numpy()
    pred_slice = prediction[0, 0, :, :, slice_idx].cpu().numpy()

    # FFT computations (2D)
    fft_noisy = np.fft.fft2(noisy_slice)
    fft_gt = np.fft.fft2(gt_slice)
    fft_pred = np.fft.fft2(pred_slice)

    # Magnitudes
    mag_noisy = np.log(np.abs(np.fft.fftshift(fft_noisy)) + 1e-8)
    mag_gt = np.log(np.abs(np.fft.fftshift(fft_gt)) + 1e-8)
    mag_pred = np.log(np.abs(np.fft.fftshift(fft_pred)) + 1e-8)

    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    # Spatial domain
    im0 = axes[0, 0].imshow(noisy_slice, cmap='gray')
    axes[0, 0].set_title("Noisy Input")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(gt_slice, cmap='gray')
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(pred_slice, cmap='gray')
    axes[0, 2].set_title("Prediction")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Difference images
    im3 = axes[1, 0].imshow(np.abs(noisy_slice - gt_slice), cmap='hot')
    axes[1, 0].set_title("Noisy vs GT Diff")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(np.abs(gt_slice - pred_slice), cmap='hot')
    axes[1, 1].set_title("GT vs Pred Diff")
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    im5 = axes[1, 2].imshow(np.abs(noisy_slice - pred_slice), cmap='hot')
    axes[1, 2].set_title("Noisy vs Pred Diff")
    axes[1, 2].axis("off")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    # Frequency domain
    im6 = axes[2, 0].imshow(mag_noisy, cmap='viridis')
    axes[2, 0].set_title("Noisy FFT Magnitude")
    axes[2, 0].axis("off")
    plt.colorbar(im6, ax=axes[2, 0], fraction=0.046)

    im7 = axes[2, 1].imshow(mag_gt, cmap='viridis')
    axes[2, 1].set_title("GT FFT Magnitude")
    axes[2, 1].axis("off")
    plt.colorbar(im7, ax=axes[2, 1], fraction=0.046)

    im8 = axes[2, 2].imshow(mag_pred, cmap='viridis')
    axes[2, 2].set_title("Pred FFT Magnitude")
    axes[2, 2].axis("off")
    plt.colorbar(im8, ax=axes[2, 2], fraction=0.046)

    plt.tight_layout()
    writer.add_figure(f"{prefix}Frequency/Full_Comparison", fig, global_step)
    plt.close(fig)






