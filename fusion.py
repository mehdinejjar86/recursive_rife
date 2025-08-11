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
      I0, I1: [N, C, H, W]
      flows:  list length K, each [N, 4, H, W] with [:,:2]=f01, [:,2:4]=f10
      masks:  list length K, each [N, 1, H, W]

    Returns:
      out:        [N, C, H, W]  blended output
      f01_comb:   [N, 2, H, W]  fused forward flow
      f10_comb:   [N, 2, H, W]  fused backward flow
      m_comb:     [N, 1, H, W]  fused mask (after sigmoid)
      weights:    [K, N, 1, H, W] per-pixel fusion weights (softmax)
    """
    def __init__(self, img_channels=3):
        super().__init__()
        in_ch = 2 * img_channels + 5
        self.score_net = ScoreNet(in_ch)

    def forward(self, I0, I1, flows, masks):
        device = I0.device
        dtype = I0.dtype

        K = len(flows)
        assert K == len(masks) and K > 0, "Need same number of flows and masks (>0)."
        print(f"I0 shape: {I0.shape}, I1 shape: {I1.shape}, flows shape: {flows.shape}, masks shape: {masks.shape}")
        if len(I0.shape) == 5:
            I0 = I0.squeeze(2)  # Remove the singleton dimension (1)
        if len(I1.shape) == 5:
            I1 = I1.squeeze(2)  # Remove the singleton dimension (1)
        
        N, C, H, W = I0.shape
        assert I1.shape == (N, C, H, W), "I0 and I1 must have the same shape."

        # Remove the singleton dimension in flows and masks
        flows = flows.squeeze(2)  # Shape: [16, 4, 2048, 2048] (after removing singleton dim)
        masks = masks.squeeze(2).squeeze(2)  # Shape: [16, 4, 2048, 2048] (after removing singleton dims)

        # Ensure flow and mask shapes are correct
        for f, m in zip(flows, masks):
            assert f.shape == (N, 4, H, W), f"flow must be [N, 4, H, W], got {f.shape}"
            assert m.shape == (N, 4, H, W), f"mask must be [N, 4, H, W], got {m.shape}"

        # Build per-candidate inputs: [I0, I1, f01, f10, m] => [N, 2C+5, H, W]
        inputs = []
        for f, m in zip(flows, masks):
            # Split the flow into forward and backward components
            f01 = f[:, :2]  # Forward flow (first 2 channels)
            f10 = f[:, 2:4]  # Backward flow (last 2 channels)
            feat = torch.cat([I0, I1, f01, f10, m], dim=1)  # [N, 2C+5, H, W]
            inputs.append(feat)

        # Run a shared ScoreNet over all candidates by stacking along batch
        x = torch.cat(inputs, dim=0)                        # [K*N, 2C+5, H, W]
        scores = self.score_net(x)                          # [K*N, 1, H, W]
        scores = scores.view(K, N, 1, H, W)                 # [K, N, 1, H, W]

        # Softmax across candidates per pixel -> weights sum to 1 at each (n, y, x)
        weights = torch.softmax(scores, dim=0)              # [K, N, 1, H, W]

        # Fuse flows with per-pixel weights
        f01_list = [f[:, :2] for f in flows]                # each [N, 2, H, W]
        f10_list = [f[:, 2:4] for f in flows]

        # Stack to [K, N, 2, H, W]
        f01 = torch.stack(f01_list, dim=0)
        f10 = torch.stack(f10_list, dim=0)

        # Broadcast weights over the 2 flow channels
        w2 = weights.expand(K, N, 2, H, W)

        f01_comb = (w2 * f01).sum(dim=0)                    # [N, 2, H, W]
        f10_comb = (w2 * f10).sum(dim=0)                    # [N, 2, H, W]

        # Warp once with fused flows
        I0w = warp(I0, f01_comb)                            # [N, C, H, W]
        I1w = warp(I1, f10_comb)                            # [N, C, H, W]

        # Fuse masks with same weights (then sigmoid)
        m = torch.stack(masks, dim=0)                       # [K, N, 1, H, W]
        m_comb_pre = (weights * m).sum(dim=0)               # [N, 1, H, W]
        m_comb = torch.sigmoid(m_comb_pre)                  # [N, 1, H, W]

        # Blend
        out = I0w * m_comb + I1w * (1 - m_comb)             # [N, C, H, W]
        return out, f01_comb, f10_comb, m_comb, weights
