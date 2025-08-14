import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Start of new, more elegant warp functions ---

def create_normalized_grid(H, W, device):
    """
    Creates a normalized sampling grid for a given image size.
    The grid ranges from -1 to 1. This is a helper function
    for the elegant_warp function.
    """
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    # The output of meshgrid is [H, W], we need [W, H] for the grid_sample
    # So we stack as (grid_x, grid_y) to get a grid of shape [1, H, W, 2]
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)


def elegant_warp(image, flow):
    """
    Warps an image using a flow field. This function uses
    torch.nn.functional.grid_sample with align_corners=False
    to ensure a stable and consistent gradient flow during backpropagation.
    
    Args:
        image (torch.Tensor): The source image to be warped, shape [N, C, H, W].
        flow (torch.Tensor): The optical flow field, shape [N, 2, H, W].
                                 The flow is in pixel coordinates.
                                                  
    Returns:
        torch.Tensor: The warped image, shape [N, C, H, W].
    """
    N, C, H, W = image.shape
    
    # Create the normalized grid once for this image size and device
    normalized_grid = create_normalized_grid(H, W, image.device)
    
    # Normalize the flow field to the range [-1, 1]
    # The flow is in pixel coordinates, so we need to scale it.
    flow_x = flow[:, 0, :, :] / ((W - 1) / 2)
    flow_y = flow[:, 1, :, :] / ((H - 1) / 2)
    
    # Reshape the normalized flow to match the grid's dimensions and stack
    normalized_flow = torch.stack((flow_x, flow_y), dim=-1) # Shape: [N, H, W, 2]
    
    # Add the normalized flow to the normalized grid
    sampling_grid = normalized_grid + normalized_flow
    
    # Conditionally set padding_mode based on the device
    padding_mode = 'zeros' if image.device.type == 'mps' else 'border'
    
    # Perform the grid sampling with align_corners=False
    return F.grid_sample(
        input=image,
        grid=sampling_grid,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=False
    )

# --- End of new, more elegant warp functions ---

class ResidualBlock(nn.Module):
    """
    A simple residual block to help with gradient flow.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out + identity)
        return out


class DynamicFusionModule(nn.Module):
    """
    An innovative module that learns to dynamically fuse features from
    multiple candidates. It takes the concatenated features and outputs
    a single, fused feature map.
    """
    def __init__(self, in_ch, out_ch, num_candidates):
        super().__init__()
        self.num_candidates = num_candidates
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # This layer is now part of the model and will be trained
        self.weight_conv = nn.Conv2d(out_ch, num_candidates, kernel_size=1)

    def forward(self, x):
        N_k, C, h, w = x.shape
        N = N_k // self.num_candidates
        
        # Process concatenated features
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Reshape features to perform weighted sum
        out_features = out.view(N, self.num_candidates, out.shape[1], h, w)

        # Learn per-pixel weights for each candidate
        weights = self.weight_conv(out)
        
        # Correctly reshape weights from [N*K, K, h, w] to [N, K, K, h, w]
        weights = weights.view(N, self.num_candidates, self.num_candidates, h, w)
        
        # Apply a softmax across the candidates to get fusion weights
        fusion_weights = F.softmax(weights, dim=1)
        
        # Broadcast fusion weights and perform weighted sum
        fused_out = (fusion_weights[:,:,0:1,:,:] * out_features).sum(dim=1)
        
        return fused_out, fusion_weights[:,:,0,:,:]


class DynamicFusionUNet(nn.Module):
    """
    A U-Net architecture with an integrated dynamic fusion module
    in the bottleneck. This allows for smarter, learned fusion of
    multiple candidates.
    """
    def __init__(self, in_ch, hid=32, num_candidates=3):
        super().__init__()
        self.hid = hid
        self.num_candidates = num_candidates
        
        # Encoder (Downsampling path)
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, hid, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(hid, hid * 2, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # The number of candidates is now passed to the forward method
        self.fusion_module = DynamicFusionModule(hid * 2, hid * 2, num_candidates)
        
        # Decoder (Upsampling path)
        self.upconv1 = nn.ConvTranspose2d(hid * 2, hid * 2, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(hid * 4, hid * 2, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.upconv2 = nn.ConvTranspose2d(hid * 2, hid, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(hid * 2, hid, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        # Final output layer - This will now produce the per-candidate weights
        self.weights_conv = nn.Conv2d(hid, num_candidates, kernel_size=1)

    def forward(self, x):
        N_k, C, H, W = x.shape
        N = N_k // self.num_candidates

        # Encoder path - Process all candidates as a single batch
        enc1_out_all = self.enc1(x)  # [N*K, hid, H, W]
        pooled1_all = self.pool1(enc1_out_all)  # [N*K, hid, H/2, W/2]
        enc2_out_all = self.enc2(pooled1_all)  # [N*K, hid*2, H/2, W/2]
        pooled2_all = self.pool2(enc2_out_all)  # [N*K, hid*2, H/4, W/4]

        # Dynamic Fusion Module in the bottleneck
        fused_bottleneck_features, fusion_weights_bottleneck = self.fusion_module(pooled2_all)
        
        # Fuse skip connection features using the same learned fusion weights
        # Upsample fusion weights to match the size of the first skip connection
        # FIX: The extra unsqueeze was causing a dimension mismatch. We interpolate
        # first, then add the dimension for broadcasting.
        weights_upsampled_to_enc2 = F.interpolate(fusion_weights_bottleneck, size=(H//2, W//2), mode='bilinear', align_corners=False).unsqueeze(2) # [N, K, 1, H/2, W/2]
        enc2_out_reshaped = enc2_out_all.view(N, self.num_candidates, self.hid*2, H//2, W//2) # [N, K, hid*2, H/2, W/2]
        fused_skip1_features = (weights_upsampled_to_enc2 * enc2_out_reshaped).sum(dim=1) # [N, hid*2, H/2, W/2]
        
        # Fuse the second skip connection
        # FIX: Same as above, interpolate first, then unsqueeze.
        weights_upsampled_to_enc1 = F.interpolate(fusion_weights_bottleneck, size=(H, W), mode='bilinear', align_corners=False).unsqueeze(2) # [N, K, 1, H, W]
        enc1_out_reshaped = enc1_out_all.view(N, self.num_candidates, self.hid, H, W) # [N, K, hid, H, W]
        fused_skip2_features = (weights_upsampled_to_enc1 * enc1_out_reshaped).sum(dim=1) # [N, hid, H, W]

        # Decoder path with fused skip connections
        up_out1 = self.upconv1(fused_bottleneck_features)
        dec1_in = torch.cat([up_out1, fused_skip1_features], dim=1)
        dec1_out = self.dec1(dec1_in)
        
        up_out2 = self.upconv2(dec1_out)
        dec2_in = torch.cat([up_out2, fused_skip2_features], dim=1)
        dec2_out = self.dec2(dec2_in)
        
        # We return the upsampled weights for the final fusion
        weights = self.weights_conv(dec2_out)
        return weights, dec2_out

class PixelFlowFusionNetwork(nn.Module):
    """
    Per-pixel fusion of K candidate flows + masks using a lightweight
    learned score network. This version uses the new DynamicFusionUNet
    for a more robust, learned fusion process.
    """
    def __init__(self, img_channels=3, num_candidates=3):
        super().__init__()
        in_ch = 2 * img_channels + 6
        self.fusion_net = DynamicFusionUNet(in_ch, num_candidates=num_candidates)
        self.num_candidates = num_candidates

    def forward(self, I0, I1, flows, masks, timesteps):
        device = I0.device
        dtype = I0.dtype

        K = flows.shape[1]
        N, C, H, W = I0.shape
        
        assert K == self.num_candidates, "Number of candidates must match the network's config."
        
        I0_expanded = I0.unsqueeze(1).expand(N, K, C, H, W).reshape(-1, C, H, W)
        I1_expanded = I1.unsqueeze(1).expand(N, K, C, H, W).reshape(-1, C, H, W)
        timesteps_expanded = timesteps.unsqueeze(-1).unsqueeze(-1).expand(N, K, H, W)
        flows_reshaped = flows.reshape(-1, 4, H, W)
        masks_reshaped = masks.unsqueeze(2).reshape(-1, 1, H, W)

        x = torch.cat([I0_expanded, I1_expanded, flows_reshaped, masks_reshaped, timesteps_expanded.reshape(-1, 1, H, W)], dim=1)
        
        # The fusion network now returns the final scores and a feature map
        scores, features = self.fusion_net(x)

        # Reshape and normalize scores to get per-pixel weights
        weights = F.softmax(scores, dim=1).unsqueeze(2)
        
        flows_stacked = flows.permute(1, 0, 2, 3, 4)
        
        # Permute weights to match flows for broadcasting
        weights_permuted = weights.permute(1, 0, 2, 3, 4)
        
        f01_stacked = flows_stacked[:,:,:2,:,:]
        f10_stacked = flows_stacked[:,:,2:4,:,:]
        
        f01_comb = (weights_permuted * f01_stacked).sum(dim=0)
        f10_comb = (weights_permuted * f10_stacked).sum(dim=0)

        I0w = elegant_warp(I0, f01_comb)
        I1w = elegant_warp(I1, f10_comb)
        
        masks = masks.permute(1, 0, 2, 3).unsqueeze(2)
        m_comb = (weights_permuted * masks).sum(dim=0)
        m_comb = torch.sigmoid(m_comb)

        out = I0w * m_comb + I1w * (1 - m_comb)
        
        return out