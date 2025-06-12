import glob
import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset



class FMRI3DDataset(Dataset):
    def __init__(self, root_dir, noise_level=0.1):
        """
        Args:
            root_dir (str): Directory where the fMRI .nii.gz files are located.
            noise_level (float): Max fraction of image intensity used to scale Gaussian noise.
        """
        self.volume_paths = glob.glob(os.path.join(root_dir, '**/*.nii.gz'), recursive=True)
        self.samples = []
        self.noise_level = noise_level
        self.volume_cache = {}

        for path in self.volume_paths:
            nii = nib.load(path)
            data = nii.get_fdata()

            if data.ndim == 4:  # Only consider 4D fMRI volumes
                for t in range(data.shape[3]):
                    self.samples.append((path, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, t = self.samples[idx]

        # Load and cache the volume
        if path not in self.volume_cache:
            self.volume_cache[path] = nib.load(path).get_fdata()

        volume = self.volume_cache[path][..., t]  # 3D volume at time t: shape (X, Y, Z)

        # Normalize volume to [0, 1]
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)

        # Add Gaussian noise slice-by-slice (Z) at this timepoint
        noisy_volume = np.zeros_like(volume)
        for z in range(volume.shape[2]):
            slice_ = volume[:, :, z]
            max_intensity = np.max(slice_)
            noise_std = np.random.uniform(0, self.noise_level) * max_intensity
            noise = np.random.normal(0, noise_std, slice_.shape)
            noisy_volume[:, :, z] = slice_ + noise

        # Convert to torch tensors and add channel dim (1, X, Y, Z)
        x_clean = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        x_noisy = torch.tensor(noisy_volume, dtype=torch.float32).unsqueeze(0)

        # Optional: Clamp to [0, 1] to stay in valid range
        x_noisy = torch.clamp(x_noisy, 0.0, 1.0)

        # Optionally clear cache if too big
        if len(self.volume_cache) > 4:
            self.volume_cache.clear()

        return x_noisy, x_clean
