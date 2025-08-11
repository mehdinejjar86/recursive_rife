import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bar
from data_fusion import FlowMaskDataset  
from fusion import PixelFlowFusionNetwork  
from model.pytorch_msssim import ssim_matlab as ssim


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

# Specify the root directory where your data folders are located
root_dir = 'dataset_fusion'

# Instantiate the dataset
dataset = FlowMaskDataset(root_dir)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Instantiate the model and move to the device
model = PixelFlowFusionNetwork(img_channels=3).to(device)

# Define SSIM Loss and Optimizer
criterion_ssim = lambda x, y: 1 - ssim(x, y, data_range=1, size_average=True)  # 1 - SSIM to make it a loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 10  # Specify the number of epochs

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    total_loss = 0  # Initialize total loss for the epoch
    total_ssim = 0  # Initialize total SSIM score for the epoch
    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100, position=0, leave=True)

    for i, (I0, I1, flows, masks, l_gt) in enumerate(loop):
        
        # Move data to the appropriate device
        I0, I1, flows, masks, l_gt = I0.to(device), I1.to(device), flows.to(device), masks.to(device), l_gt.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        out, f01_comb, f10_comb, m_comb, weights = model(I0, I1, flows, masks)

        # Compute SSIM loss between the model output and ground truth
        loss_ssim = criterion_ssim(out, l_gt)  # Use SSIM loss
        total_loss += loss_ssim.item()

        # Backward pass and optimization
        loss_ssim.backward()
        optimizer.step()

        # Accumulate SSIM score (1 - SSIM, since we want to minimize)
        ssim_value = ssim(out, l_gt, data_range=1, size_average=True)
        total_ssim += ssim_value.item()

        # Update progress bar description with the current loss and SSIM
        loop.set_postfix(loss=loss_ssim.item(), ssim=ssim_value.item())

    # Print total loss and average SSIM for the epoch
    avg_loss = total_loss / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Average SSIM: {avg_ssim:.4f}")

    # Optional: Save the model after every epoch
    # torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    
    # Optional: Evaluate on validation data (if you have a validation set)
    # model.eval()  # Switch to evaluation mode
    # with torch.no_grad():
    #     # Perform validation and calculate validation loss/SSIM
    #     model.train()  # Switch back to training mode
