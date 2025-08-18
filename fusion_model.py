import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ConvBlock(nn.Module):
    """Basic convolutional block with optional normalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 norm='none', activation='relu', bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # Normalization
        self.norm = None
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(min(32, out_channels//4), out_channels)
        
        # Activation
        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with feature fusion"""
    def __init__(self, channels, norm='gn'):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, norm=norm, activation='leaky')
        self.conv2 = ConvBlock(channels, channels, norm=norm, activation='none')
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return self.activation(x)


class SpatialAttention(nn.Module):
    """Spatial attention module for feature weighting"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module using squeeze-and-excitation"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for fusing features from multiple anchors"""
    def __init__(self, channels, num_heads=4, downsample_factor=8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.downsample_factor = downsample_factor

        self.q_conv = nn.Conv2d(channels, channels, 1)
        self.k_conv = nn.Conv2d(channels, channels, 1)
        self.v_conv = nn.Conv2d(channels, channels, 1)
        self.out_conv = nn.Conv2d(channels, channels, 1)

        # Added downsampling and upsampling layers
        self.downsample = nn.MaxPool2d(kernel_size=downsample_factor)
        self.upsample = nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False)

        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, keys, values):
        """
        Args:
            query: [B, C, H, W] - query features
            keys: [B, N, C, H, W] - key features from N anchors
            values: [B, N, C, H, W] - value features from N anchors
        """
        B, C, H, W = query.shape
        N = keys.shape[1]
        
        # Downsample features before attention
        q_down = self.downsample(query)
        keys_down = self.downsample(keys.view(B*N, C, H, W)).view(B, N, C, H//self.downsample_factor, W//self.downsample_factor)
        values_down = self.downsample(values.view(B*N, C, H, W)).view(B, N, C, H//self.downsample_factor, W//self.downsample_factor)
        
        H_down, W_down = q_down.shape[-2:]

        # Compute Q on downsampled features
        q = self.q_conv(q_down)
        q = q.view(B, self.num_heads, self.head_dim, H_down * W_down).transpose(-2, -1)
        
        # Compute K and V for all anchors on downsampled features
        all_attention = []
        for i in range(N):
            k = self.k_conv(keys_down[:, i])
            v = self.v_conv(values_down[:, i])
            
            k = k.view(B, self.num_heads, self.head_dim, H_down * W_down)
            v = v.view(B, self.num_heads, self.head_dim, H_down * W_down).transpose(-2, -1)
            
            # Compute attention scores
            scores = torch.matmul(q, k) * self.scale
            attention = F.softmax(scores, dim=-1)
            
            # Apply attention to values
            out = torch.matmul(attention, v)
            all_attention.append(out)
        
        # Combine and reshape
        combined = torch.stack(all_attention, dim=1).mean(dim=1)
        combined = combined.transpose(-2, -1).contiguous().view(B, C, H_down, W_down)
        
        # Upsample back to original resolution and apply final convolution
        combined = self.upsample(combined)
        return self.out_conv(combined)


class TemporalWeightingModule(nn.Module):
    """Module to compute importance weights based on temporal distance"""
    def __init__(self, num_anchors=3, hidden_dim=128):
        super().__init__()
        self.num_anchors = num_anchors
        
        # Network to predict weights from timesteps
        self.weight_net = nn.Sequential(
            nn.Linear(num_anchors, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_anchors),
            nn.Softmax(dim=1)
        )
    
    def forward(self, timesteps):
        """
        Args:
            timesteps: [B, num_anchors] - temporal distances for each anchor
        Returns:
            weights: [B, num_anchors] - importance weights
        """
        return self.weight_net(timesteps)


class FlowWarping(nn.Module):
    """Differentiable flow warping module"""
    def forward(self, img, flow):
        """
        Args:
            img: [B, C, H, W]
            flow: [B, 2, H, W] - optical flow
        """
        B, C, H, W = img.size()
        
        # Create coordinate grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(img.device)
        
        # Add flow to grid
        vgrid = grid + flow
        
        # Normalize to [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        
        # Warp image
        output = F.grid_sample(img, vgrid, align_corners=True)
        return output


class MultiAnchorFusionModel(nn.Module):
    """
    Novel multi-anchor fusion model for video frame interpolation
    Fuses information from multiple anchor pairs with different temporal distances
    """
    def __init__(self, num_anchors=3, base_channels=64, downsample_factor=8, use_deformable=False):
        super().__init__()
        self.num_anchors = num_anchors
        self.base_channels = base_channels
        self.use_deformable = use_deformable
        
        # Flow warping module
        self.flow_warp = FlowWarping()
        
        # Temporal weighting module
        self.temporal_weighter = TemporalWeightingModule(num_anchors)
        
        # Shared feature extractor for efficiency
        self.shared_encoder = self._make_shared_encoder()
        
        # Anchor-specific adaptation layers
        self.anchor_adapters = nn.ModuleList([
            nn.Conv2d(base_channels * 2, base_channels * 2, 1) for _ in range(num_anchors)
        ])
        
        # Flow refinement networks for each anchor
        self.flow_refiners = nn.ModuleList([
            self._make_flow_refiner() for _ in range(num_anchors)
        ])
        
        # Mask refinement networks
        self.mask_refiners = nn.ModuleList([
            self._make_mask_refiner() for _ in range(num_anchors)
        ])
        
        # Cross-attention fusion modules
        self.cross_attention_low = CrossAttentionFusion(base_channels * 2, downsample_factor=downsample_factor)
        self.cross_attention_mid = CrossAttentionFusion(base_channels * 4, downsample_factor=downsample_factor)
        self.cross_attention_high = CrossAttentionFusion(base_channels * 8, downsample_factor=downsample_factor)
        
        # Hierarchical fusion decoder
        self.decoder = self._make_decoder()
        
        # Context aggregation module
        # Each anchor contributes base_channels*2 features from low level
        self.context_aggregator = nn.Sequential(
            ConvBlock(base_channels * 2 * num_anchors, base_channels * 2, 1, norm='gn', activation='leaky'),
            ConvBlock(base_channels * 2, base_channels, 3, 1, 1, norm='gn', activation='leaky')
        )
        
        # Final synthesis network
        self.synthesis = nn.Sequential(
            ConvBlock(base_channels + 3, base_channels, norm='gn', activation='leaky'),
            ResidualBlock(base_channels, norm='gn'),
            ResidualBlock(base_channels, norm='gn'),
            ConvBlock(base_channels, base_channels//2, norm='gn', activation='leaky'),
            ConvBlock(base_channels//2, 3, activation='sigmoid')  # Use sigmoid for [0,1] range
        )
        
        # Residual prediction head
        self.residual_head = nn.Sequential(
            ConvBlock(base_channels, base_channels//2, norm='gn', activation='leaky'),
            ConvBlock(base_channels//2, 3, activation='tanh')  # Residual can be negative
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_shared_encoder(self):
        """Create shared feature encoder"""
        return nn.ModuleDict({
            'low': nn.Sequential(
                ConvBlock(11, self.base_channels, 7, 1, 3, norm='none', activation='leaky'),
                ConvBlock(self.base_channels, self.base_channels * 2, 3, 2, 1, norm='gn', activation='leaky'),
                ResidualBlock(self.base_channels * 2, norm='gn')
            ),
            'mid': nn.Sequential(
                ConvBlock(self.base_channels * 2, self.base_channels * 4, 3, 2, 1, norm='gn', activation='leaky'),
                ResidualBlock(self.base_channels * 4, norm='gn'),
                ResidualBlock(self.base_channels * 4, norm='gn')
            ),
            'high': nn.Sequential(
                ConvBlock(self.base_channels * 4, self.base_channels * 8, 3, 2, 1, norm='gn', activation='leaky'),
                ResidualBlock(self.base_channels * 8, norm='gn'),
                ResidualBlock(self.base_channels * 8, norm='gn'),
                ResidualBlock(self.base_channels * 8, norm='gn')
            )
        })
    
    def _make_flow_refiner(self):
        """Create flow refinement network"""
        return nn.Sequential(
            ConvBlock(4, self.base_channels, 5, 1, 2, norm='none', activation='leaky'),
            ResidualBlock(self.base_channels, norm='gn'),
            ConvBlock(self.base_channels, self.base_channels//2, norm='gn', activation='leaky'),
            ConvBlock(self.base_channels//2, 4, activation='none')  # Output refined flow (2 channels each for forward/backward)
        )
    
    def _make_mask_refiner(self):
        """Create mask refinement network"""
        return nn.Sequential(
            ConvBlock(1, self.base_channels//2, 5, 1, 2, norm='none', activation='leaky'),
            ResidualBlock(self.base_channels//2, norm='gn'),
            ConvBlock(self.base_channels//2, 1, activation='sigmoid')
        )
    
    def _make_decoder(self):
        """Create hierarchical decoder"""
        return nn.ModuleDict({
            'up_high_to_mid': nn.Sequential(
                ConvBlock(self.base_channels * 8, self.base_channels * 4, norm='gn', activation='leaky')
            ),
            'fuse_mid': nn.Sequential(
                ConvBlock(self.base_channels * 8, self.base_channels * 4, norm='gn', activation='leaky'),
                ResidualBlock(self.base_channels * 4, norm='gn'),
                SpatialAttention(self.base_channels * 4)
            ),
            'up_mid_to_low': nn.Sequential(
                ConvBlock(self.base_channels * 4, self.base_channels * 2, norm='gn', activation='leaky')
            ),
            'fuse_low': nn.Sequential(
                ConvBlock(self.base_channels * 4, self.base_channels * 2, norm='gn', activation='leaky'),
                ResidualBlock(self.base_channels * 2, norm='gn'),
                SpatialAttention(self.base_channels * 2)
            ),
            'up_to_original': nn.Sequential(
                ConvBlock(self.base_channels * 2, self.base_channels, norm='gn', activation='leaky'),
                ChannelAttention(self.base_channels)
            )
        })
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, I0_all, I1_all, flows_all, masks_all, timesteps):
        """
        Args:
            I0_all: [B, num_anchors, 3, H, W] - left anchor frames
            I1_all: [B, num_anchors, 3, H, W] - right anchor frames
            flows_all: [B, num_anchors, 4, H, W] - optical flows (2 forward, 2 backward)
            masks_all: [B, num_anchors, 1, H, W] - occlusion masks
            timesteps: [B, num_anchors] - temporal distances
        
        Returns:
            output: [B, 3, H, W] - synthesized frame
            aux_outputs: dict with intermediate results for loss computation
        """
        B, N, _, H, W = I0_all.shape
        device = I0_all.device
        
        # Get temporal weights
        temporal_weights = self.temporal_weighter(timesteps)  # [B, N]
        
        # Process each anchor pair
        warped_imgs = []
        refined_masks = []
        anchor_features = {'low': [], 'mid': [], 'high': []}
        context_features = []
        
        for i in range(N):
            I0 = I0_all[:, i]  # [B, 3, H, W]
            I1 = I1_all[:, i]  # [B, 3, H, W]
            flow = flows_all[:, i]  # [B, 4, H, W]
            mask = masks_all[:, i].unsqueeze(1) if masks_all[:, i].dim() == 3 else masks_all[:, i]  # [B, 1, H, W]
            
            # Refine flow
            refined_flow = flow + self.flow_refiners[i](flow)
            flow_01 = refined_flow[:, :2]  # Forward flow
            flow_10 = refined_flow[:, 2:]  # Backward flow
            
            # Warp images
            warped_I0 = self.flow_warp(I0, flow_01)
            warped_I1 = self.flow_warp(I1, flow_10)
            
            # Refine mask
            refined_mask = self.mask_refiners[i](mask)
            
            # Weighted combination of warped images
            warped = warped_I0 * refined_mask + warped_I1 * (1 - refined_mask)
            warped_imgs.append(warped)
            refined_masks.append(refined_mask)
            
            # Extract hierarchical features using shared encoder
            # Concatenate: [I0, I1, warped, flow_01]
            anchor_input = torch.cat([I0, I1, warped, flow_01], dim=1)  # [B, 11, H, W]
            
            # Encode features at multiple scales
            low_feat = self.shared_encoder['low'](anchor_input)
            # Apply anchor-specific adaptation
            low_feat = self.anchor_adapters[i](low_feat)
            
            mid_feat = self.shared_encoder['mid'](low_feat)
            high_feat = self.shared_encoder['high'](mid_feat)
            
            # Apply temporal weighting
            weight = temporal_weights[:, i:i+1, None, None]
            anchor_features['low'].append(low_feat * weight)
            anchor_features['mid'].append(mid_feat * weight)
            anchor_features['high'].append(high_feat * weight)
            
            # Store context features
            context_features.append(low_feat)
        
        # Stack features for cross-attention
        low_features = torch.stack(anchor_features['low'], dim=1)  # [B, N, C, H/2, W/2]
        mid_features = torch.stack(anchor_features['mid'], dim=1)  # [B, N, C, H/4, W/4]
        high_features = torch.stack(anchor_features['high'], dim=1)  # [B, N, C, H/8, W/8]
        
        # Apply cross-attention fusion at each scale
        # Use weighted mean as query
        query_high = high_features.mean(dim=1)
        fused_high = self.cross_attention_high(query_high, high_features, high_features)
        
        # Decode hierarchically
        # High to mid
        up_high = F.interpolate(fused_high, size=mid_features.shape[-2:], mode='bilinear', align_corners=True)
        up_high = self.decoder['up_high_to_mid'](up_high)
        query_mid = mid_features.mean(dim=1)
        fused_mid = self.cross_attention_mid(query_mid, mid_features, mid_features)
        fused_mid = self.decoder['fuse_mid'](torch.cat([up_high, fused_mid], dim=1))
        
        # Mid to low
        up_mid = F.interpolate(fused_mid, size=low_features.shape[-2:], mode='bilinear', align_corners=True)
        up_mid = self.decoder['up_mid_to_low'](up_mid)
        query_low = low_features.mean(dim=1)
        fused_low = self.cross_attention_low(query_low, low_features, low_features)
        fused_low = self.decoder['fuse_low'](torch.cat([up_mid, fused_low], dim=1))
        
        # Low to original resolution
        decoded_features = F.interpolate(fused_low, size=(H, W), mode='bilinear', align_corners=True)
        decoded_features = self.decoder['up_to_original'](decoded_features)
        
        # Aggregate context features
        context_concat = torch.cat(context_features, dim=1)  # [B, N*C, H/2, W/2]
        # Ensure size matches exactly with decoded_features
        context_upsampled = F.interpolate(context_concat, size=(H, W), mode='bilinear', align_corners=True)
        context_aggregated = self.context_aggregator(context_upsampled)
        
        # Combine decoded features with context - ensure same size
        if decoded_features.shape != context_aggregated.shape:
            # If there's a size mismatch, interpolate to match
            context_aggregated = F.interpolate(context_aggregated, size=decoded_features.shape[-2:], mode='bilinear', align_corners=True)
        
        decoded_features = decoded_features + context_aggregated
        
        # Compute weighted average of warped images
        warped_stack = torch.stack(warped_imgs, dim=1)  # [B, N, 3, H, W]
        weights_expanded = temporal_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1, 1]
        warped_avg = (warped_stack * weights_expanded).sum(dim=1)  # [B, 3, H, W]
        
        # Final synthesis
        synthesis_input = torch.cat([decoded_features, warped_avg], dim=1)
        synthesized = self.synthesis(synthesis_input)
        
        # Add residual connection
        residual = self.residual_head(decoded_features) * 0.1  # Scale down residual
        output = synthesized + residual
        output = torch.clamp(output, 0, 1)  # Ensure output is in [0, 1]
        
        # Prepare auxiliary outputs for loss computation
        aux_outputs = {
            'warped_imgs': warped_imgs,
            'refined_masks': refined_masks,
            'temporal_weights': temporal_weights,
            'warped_avg': warped_avg,
            'residual': residual,
            'synthesized': synthesized
        }
        
        return output, aux_outputs


class FusionLoss(nn.Module):
    """Loss function for multi-anchor fusion model"""
    def __init__(self, lambda_l1=1.0, lambda_perceptual=0.1, lambda_smooth=0.01, lambda_consistency=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_smooth = lambda_smooth
        self.lambda_consistency = lambda_consistency
        
        # VGG features for perceptual loss (simplified)
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def perceptual_loss(self, pred, target):
        """Simplified perceptual loss using normalized features"""
        # Normalize
        pred_norm = (pred - self.vgg_mean) / self.vgg_std
        target_norm = (target - self.vgg_mean) / self.vgg_std
        
        # Multi-scale L1 loss as proxy for perceptual loss
        loss = 0
        for scale in [1, 0.5, 0.25]:
            if scale != 1:
                pred_scaled = F.interpolate(pred_norm, scale_factor=scale, mode='bilinear', align_corners=True)
                target_scaled = F.interpolate(target_norm, scale_factor=scale, mode='bilinear', align_corners=True)
            else:
                pred_scaled = pred_norm
                target_scaled = target_norm
            
            loss += F.l1_loss(pred_scaled, target_scaled)
        
        return loss / 3
    
    def smoothness_loss(self, flow):
        """TV-L1 smoothness loss for flow"""
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        return (dx.mean() + dy.mean()) / 2
    
    def consistency_loss(self, warped_imgs, target):
        """Consistency loss between warped images and target"""
        loss = 0
        for warped in warped_imgs:
            loss += F.l1_loss(warped, target)
        return loss / len(warped_imgs)
    
    def forward(self, output, target, aux_outputs):
        """
        Compute total loss
        
        Args:
            output: [B, 3, H, W] - predicted frame
            target: [B, 3, H, W] - ground truth frame
            aux_outputs: dict with auxiliary outputs
        """
        # L1 reconstruction loss
        l1_loss = F.l1_loss(output, target)
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(output, target)
        
        # Consistency loss on warped images
        consistency_loss = self.consistency_loss(aux_outputs['warped_imgs'], target)
        
        # Total loss
        total_loss = (self.lambda_l1 * l1_loss + 
                     self.lambda_perceptual * perceptual_loss +
                     self.lambda_consistency * consistency_loss)
        
        # Return losses for logging
        return {
            'total': total_loss,
            'l1': l1_loss,
            'perceptual': perceptual_loss,
            'consistency': consistency_loss
        }


def create_fusion_model(num_anchors=3, base_channels=64, downsample_factor=8):
    """Factory function to create the fusion model"""
    return MultiAnchorFusionModel(num_anchors=num_anchors, base_channels=base_channels, downsample_factor=downsample_factor)


# Example usage
if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_fusion_model(num_anchors=3, base_channels=64).to(device)
    
    # Create dummy inputs
    batch_size = 2
    num_anchors = 3
    height, width = 2048, 2048
    
    I0_all = torch.randn(batch_size, num_anchors, 3, height, width).to(device)
    I1_all = torch.randn(batch_size, num_anchors, 3, height, width).to(device)
    flows_all = torch.randn(batch_size, num_anchors, 4, height, width).to(device)
    masks_all = torch.rand(batch_size, num_anchors, height, width).to(device)
    timesteps = torch.rand(batch_size, num_anchors).to(device)
    
    # Forward pass
    output, aux_outputs = model(I0_all, I1_all, flows_all, masks_all, timesteps)
    
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test loss
    loss_fn = FusionLoss()
    target = torch.randn(batch_size, 3, height, width).to(device)
    losses = loss_fn(output, target, aux_outputs)
    
    print(f"\nLosses:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")