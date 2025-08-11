import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bar
from data_fusion import FlowMaskDataset  # Custom dataset
from fusion import PixelFlowFusionNetwork  # The fusion model
from model.pytorch_msssim import ssim_matlab as ssim
import os # We'll need this to set the environment variable
import torch.nn.functional as F # Needed for interpolation
from datetime import datetime
import csv

SSIM_TARGET_SIZE = (32, 32)  # Size to which images will be resized for SSIM calculation

# Device setup
# This finds the best available device: CUDA, MPS (for Apple Silicon), or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

# If the device is MPS, we enable a CPU fallback for unsupported operations
if device.type == 'mps':
    print("Detected MPS device. Setting PYTORCH_ENABLE_MPS_FALLBACK=1 to handle unsupported ops.")
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Specify the root directory where your data folders are located
root_dir = 'dataset_fusion'

# Instantiate the dataset and dataloader
dataset = FlowMaskDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Instantiate the model and move it to the selected device
model = PixelFlowFusionNetwork(img_channels=3).to(device)

# Define SSIM Loss and Optimizer
# The SSIM loss is defined as 1 - SSIM score, as we want to minimize it.
criterion_ssim = lambda x, y: 1 - ssim(x, y, size_average=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 10

# --- NEW: Setup for saving models and metrics ---
# Create a unique directory for this training run with a timestamp
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_dir = os.path.join("train", timestamp)
os.makedirs(save_dir, exist_ok=True)
print(f"Saving models and metrics to: {save_dir}")

# Initialize tracking variables
best_loss = float('inf')
metrics_file_path = os.path.join(save_dir, "metrics.csv")

# Open the metrics file and write the header
with open(metrics_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Average_Loss', 'Average_SSIM'])
# --- END NEW ---

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    
    total_loss = 0
    total_ssim = 0
    
    # Initialize the progress bar with a descriptive title
    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100, position=0, leave=True)

    for i, (I0, I1, flows, masks, I_gt) in enumerate(loop):
        # Move all tensors to the correct device in one step
        I0, I1, flows, masks, I_gt = (
            I0.to(device), I1.to(device), flows.to(device), masks.to(device), I_gt.to(device)
        )

        optimizer.zero_grad()  # Reset gradients

        # Forward pass: get the model's prediction
        out, f01_comb, f10_comb, m_comb, weights = model(I0, I1, flows, masks)

        # Rescale the output and ground truth to a smaller size for a faster SSIM calculation
        out_resized = F.interpolate(out, size=SSIM_TARGET_SIZE, mode='bilinear', align_corners=False)
        I_gt_resized = F.interpolate(I_gt.squeeze(1), size=SSIM_TARGET_SIZE, mode='bilinear', align_corners=False)

        # Compute the SSIM score once on the smaller tensors
        ssim_score = ssim(out_resized, I_gt_resized, size_average=True)
        
        # Calculate the SSIM loss (1 - SSIM score)
        ssim_loss = 1 - ssim_score
        
        total_loss += ssim_loss.item()
        total_ssim += ssim_score.item()

        # Backward pass and optimization step
        ssim_loss.backward()
        optimizer.step()

        # Update the progress bar with the current loss and SSIM score
        loop.set_postfix(loss=ssim_loss.item(), ssim=ssim_score.item())

    # Calculate and print average metrics for the epoch
    avg_loss = total_loss / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Average SSIM: {avg_ssim:.4f}")

    # --- NEW: Save model checkpoints and metrics ---
    # Save metrics to the CSV file
    with open(metrics_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch + 1, avg_loss, avg_ssim])

    # Save the best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with loss: {best_loss:.4f}")

    # Save the last model at the end of each epoch
    last_model_path = os.path.join(save_dir, "last_model.pth")
    torch.save(model.state_dict(), last_model_path)
    # --- END NEW ---
