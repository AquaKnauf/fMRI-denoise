import gradio as gr
import nibabel as nib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from unet.model import UNet3DfMRI

# Download model weights from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="AquaKnauf/fmri-denoise",
    filename="model_weights_epoch6.pth"
)

# Load model
device = torch.device("cpu")
model = UNet3DfMRI(in_channels=1, out_channels=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def denoise_fmri(file_obj):
    # Load NIfTI image
    img = nib.load(file_obj.name)
    data = img.get_fdata()

    # Normalize and prepare tensor
    input_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().to(device)  # Shape: [1, 1, D, H, W]

    with torch.no_grad():
        output = model(input_tensor)
    
    output_data = output.squeeze().cpu().numpy()

    # Save result
    denoised_img = nib.Nifti1Image(output_data, affine=img.affine, header=img.header)
    output_path = "denoised_output.nii.gz"
    nib.save(denoised_img, output_path)

    return output_path

iface = gr.Interface(
    fn=denoise_fmri,
    inputs=gr.File(label="Upload fMRI NIfTI (.nii.gz)"),
    outputs=gr.File(label="Denoised Output"),
    title="fMRI Denoising with UNet",
    description="Upload a noisy fMRI scan (.nii.gz), and the UNet model will return a denoised version."
)

iface.launch()
