import glob
import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class FMRI3DDataset(Dataset):
    def __init__(self, root_dir, noise_std=0.1):
        """
        Args:
            root_dir (str): Directory where the fMRI .nii.gz files are located.
            noise_std (float): Max std. dev. of Gaussian noise added to the input.
        """
        self.volume_paths = glob.glob(os.path.join(root_dir, '**/*.nii.gz'), recursive=True)
        self.noise_std = noise_std
        self.samples = []  # list of (path, time_idx)

        for path in self.volume_paths:
            nii = nib.load(path)
            data = nii.get_fdata()
            if data.ndim == 4:
                for t in range(data.shape[3]):
                    self.samples.append((path, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, t = self.samples[idx]

        # Load the 3D volume at time index t
        volume = nib.load(path).get_fdata()[..., t]

        # Normalize to [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

         # Add Gaussian noise per slice
        global_max = np.max(volume) 
        noisy_volume = np.zeros_like(volume)
        for z in range(volume.shape[2]):
            slice_ = volume[:, :, z]
            max_intensity = np.max(slice_)
            noise_std = np.random.uniform(0, self.noise_std) * max_intensity
            noise = np.random.normal(0, noise_std, slice_.shape)
            noisy_volume[:, :, z] = slice_ + noise

        # Convert to tensors with channel dimension
        x_clean = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        x_noisy = torch.tensor(noisy_volume, dtype=torch.float32).unsqueeze(0)

        # Clamp to [0, 1] after noise
        x_noisy = torch.clamp(x_noisy, 0.0, 1.0)


        return x_noisy, x_clean
