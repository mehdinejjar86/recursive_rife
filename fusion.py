import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp


class ScoreNet(nn.Module):
    """
    Predicts a per-pixel score map from [I0, I1, f01, f10, m].
    Input channels = 2*C + 5  (I0 & I1 are C each, flows are 4, mask is 1)
    Output: 1-channel score map (no sigmoid/softmax here).
    """
    def __init__(self, in_ch, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, hid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class PixelFlowFusionNetwork(nn.Module):
    """
    Per-pixel fusion of K candidate flows + masks using learned score maps.

    Inputs:
      I0, I1: [N, 1, C, H, W]
      flows:  [N, K, 1, 4, H, W] with [:,:2]=f01, [:,2:4]=f10
      masks:  [N, K, 1, 1, H, W]

    Returns:
      out:       [N, C, H, W]  blended output
      f01_comb:  [N, 2, H, W]  fused forward flow
      f10_comb:  [N, 2, H, W]  fused backward flow
      m_comb:    [N, 1, H, W]  fused mask (after sigmoid)
      weights:   [K, N, 1, H, W] per-pixel fusion weights (softmax)
    """
    def __init__(self, img_channels=3):
        super().__init__()
        in_ch = 2 * img_channels + 5  # For I0, I1 (C), f01, f10, m
        self.score_net = ScoreNet(in_ch)

    def forward(self, I0, I1, flows, masks):
        device = I0.device
        dtype = I0.dtype

        K = flows.shape[1]  # Number of flow candidates (K)
        N, _, C, H, W = I0.shape  # Get the shape of I0 (N, 1, C, H, W)
        assert I1.shape == (N, 1, C, H, W), "I0 and I1 must have the same shape."

        # Squeeze the singleton dimension (second dimension) of I0 and I1 to match [N, C, H, W]
        I0 = I0.squeeze(1)  # Shape: [N, C, H, W]
        I1 = I1.squeeze(1)  # Shape: [N, C, H, W]

        # Ensure correct dimensions for flows and masks
        flows = flows.squeeze(2)  # Shape: [N, K, 4, H, W]
        masks = masks.squeeze(2).squeeze(2)  # Shape: [N, K, H, W]

        # The problem is here. The permute operation should be removed.
        # masks = masks.permute(1, 0, 2, 3)  # Shape: [K, N, H, W] -> This permute is the source of the error.

        # Build per-candidate inputs: [I0, I1, f01, f10, m] => [N, 2C+5, H, W]
        inputs = []
        for k in range(K):
            f = flows[:, k, :, :, :]  # Shape: [N, 4, H, W]
            # Corrected line: extract the k-th mask for all batches
            m = masks[:, k, :, :] # Shape: [N, H, W]
            
            f01 = f[:, :2]  # Forward flow (first 2 channels)
            f10 = f[:, 2:4]  # Backward flow (last 2 channels)

            m = m.unsqueeze(1)  # Now mask has shape [N, 1, H, W]

            feat = torch.cat([I0, I1, f01, f10, m], dim=1)  # [N, 2C+5, H, W]
            inputs.append(feat)

        # Stack inputs and run a shared ScoreNet over all candidates by stacking along batch
        x = torch.cat(inputs, dim=0)  # [K*N, 2C+5, H, W]
        scores = self.score_net(x)  # [K*N, 1, H, W]
        scores = scores.view(K, N, 1, H, W)  # [K, N, 1, H, W]

        # Softmax across candidates per pixel -> weights sum to 1 at each (n, y, x)
        weights = torch.softmax(scores, dim=0)  # [K, N, 1, H, W]

        # Split flows into forward and backward components
        f01_list = [f[:, :2] for f in flows]  # This is a list of K tensors, each with shape [N, 4, H, W], so it needs to be fixed.
        f10_list = [f[:, 2:4] for f in flows]

        # Correct way to stack flows and masks for fusion
        flows_stacked = flows.permute(1, 0, 2, 3, 4) # [K, N, 4, H, W]
        f01_stacked = flows_stacked[:,:,:2,:,:]
        f10_stacked = flows_stacked[:,:,2:4,:,:]

        # Broadcast weights over the 2 flow channels
        w2 = weights.expand(K, N, 2, H, W)

        # Fuse flows with per-pixel weights
        f01_comb = (w2 * f01_stacked).sum(dim=0)  # [N, 2, H, W]
        f10_comb = (w2 * f10_stacked).sum(dim=0)  # [N, 2, H, W]

        # Warp once with fused flows
        I0w = warp(I0, f01_comb)  # [N, C, H, W]
        I1w = warp(I1, f10_comb)  # [N, C, H, W]
        
        # Reshape masks for fusion
        masks = masks.permute(1, 0, 2, 3).unsqueeze(2) # [K, N, 1, H, W]
        
        # Fuse masks with same weights (then sigmoid)
        m_comb_pre = (weights * masks).sum(dim=0)  # [N, 1, H, W]
        m_comb = torch.sigmoid(m_comb_pre)  # [N, 1, H, W]

        # Blend the warped images with the fused mask
        out = I0w * m_comb + I1w * (1 - m_comb)  # [N, 3, H, W]
        
        return out, f01_comb, f10_comb, m_comb, weights