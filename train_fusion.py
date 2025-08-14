import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from tqdm import tqdm
from data_fusion import FlowMaskDataset
from fusion import PixelFlowFusionNetwork
from model.pytorch_msssim import ssim_matlab as ssim
import torch.nn.functional as F
from datetime import datetime
import csv
from sklearn.model_selection import train_test_split
from math import log10

SSIM_TARGET_SIZE = (256, 256)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
if device.type == 'mps':
    print("Detected MPS device. Setting PYTORCH_ENABLE_MPS_FALLBACK=1 to handle unsupported ops.")
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

root_dir = './dataset_fusion'
folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
train_folders, val_folders = train_test_split(folders, test_size=0.2)
train_dataset = FlowMaskDataset(train_folders)
val_dataset = FlowMaskDataset(val_folders)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Assuming the fusion model is saved as fusion.py in the same directory
model = PixelFlowFusionNetwork(img_channels=3, num_candidates=4).to(device)

# The SSIM loss is defined as 1 - SSIM score
criterion_ssim = lambda x, y: 1 - ssim(x, y, size_average=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3)

num_epochs = 30

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_dir = os.path.join("train", timestamp)
os.makedirs(save_dir, exist_ok=True)
print(f"Saving models and metrics to: {save_dir}")

best_val_loss = float('inf')
metrics_file_path = os.path.join(save_dir, "metrics.csv")

with open(metrics_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'LR', 'Train_Loss', 'Train_SSIM_Score', 'Train_MSE_Loss', 'Val_Loss', 'Val_SSIM_Score', 'Val_MSE_Loss'])

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_train_ssim_score = 0
    total_train_mse_loss = 0
    
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", ncols=150, position=0)
    for i, (I0, I1, flows, masks, timestep, I_gt) in enumerate(loop):
        I0, I1, flows, masks, timestep, I_gt = (
            I0.to(device), I1.to(device), flows.to(device), masks.to(device), timestep.to(device), I_gt.to(device)
        )
        optimizer.zero_grad()
        out = model(I0, I1, flows, masks, timestep)

        out_resized = F.interpolate(out, size=SSIM_TARGET_SIZE, mode='bilinear', align_corners=False)
        I_gt_resized = F.interpolate(I_gt.squeeze(1), size=SSIM_TARGET_SIZE, mode='bilinear', align_corners=False)

        ssim_score = ssim(out_resized, I_gt_resized, size_average=True)
        ssim_loss = 1 - ssim_score
        mse_loss = F.mse_loss(out_resized, I_gt_resized)
        
        total_loss = ssim_loss + mse_loss
        
        total_train_loss += total_loss.item()
        total_train_ssim_score += ssim_score.item()
        total_train_mse_loss += mse_loss.item()
        
        total_loss.backward()
        optimizer.step()

        # Correctly format the output to show 6 digits after the decimal
        loop.set_postfix_str(f'loss={total_loss.item():.6f}, ssim={ssim_score.item():.6f}, mse={mse_loss.item():.6f}')
        
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_ssim_score = total_train_ssim_score / len(train_dataloader)
    avg_train_mse_loss = total_train_mse_loss / len(train_dataloader)

    model.eval()
    total_val_loss = 0
    total_val_ssim_score = 0
    total_val_mse_loss = 0

    with torch.no_grad():
        val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", ncols=150, position=1)
        for i, (I0, I1, flows, masks, timestep, I_gt) in enumerate(val_loop):
            I0, I1, flows, masks, timestep, I_gt = (
                I0.to(device), I1.to(device), flows.to(device), masks.to(device), timestep.to(device), I_gt.to(device)
            )

            out = model(I0, I1, flows, masks, timestep)
            out_resized = F.interpolate(out, size=SSIM_TARGET_SIZE, mode='bilinear', align_corners=False)
            I_gt_resized = F.interpolate(I_gt.squeeze(1), size=SSIM_TARGET_SIZE, mode='bilinear', align_corners=False)
            
            ssim_score = ssim(out_resized, I_gt_resized, size_average=True)
            ssim_loss = 1 - ssim_score
            mse_loss = F.mse_loss(out_resized, I_gt_resized)

            total_loss = ssim_loss + mse_loss
            
            total_val_loss += total_loss.item()
            total_val_ssim_score += ssim_score.item()
            total_val_mse_loss += mse_loss.item()
            
            # Correctly format the output to show 6 digits after the decimal
            val_loop.set_postfix_str(f'loss={total_loss.item():.6f}, ssim={ssim_score.item():.6f}, mse={mse_loss.item():.6f}')

    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_ssim_score = total_val_ssim_score / len(val_dataloader)
    avg_val_mse_loss = total_val_mse_loss / len(val_dataloader)

    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch + 1}/{num_epochs}], LR: {current_lr:.6f}, Train Loss: {avg_train_loss:.4f}, Train SSIM: {avg_train_ssim_score:.4f}, Train MSE: {avg_train_mse_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val SSIM: {avg_val_ssim_score:.4f}, Val MSE: {avg_val_mse_loss:.4f}")

    with open(metrics_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch + 1, current_lr, avg_train_loss, avg_train_ssim_score, avg_train_mse_loss, avg_val_loss, avg_val_ssim_score, avg_val_mse_loss])

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model with validation total loss: {best_val_loss:.4f}")

    last_model_path = os.path.join(save_dir, "last_model.pth")
    torch.save(model.state_dict(), last_model_path)
