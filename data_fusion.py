import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FlowMaskDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory containing the data folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        folder_path = os.path.join(self.root_dir, folder)

        # Load I0 and I1
        I0 = np.load(os.path.join(folder_path, 'I0.npy'))
        I1 = np.load(os.path.join(folder_path, 'I1.npy'))

        # Load flows and masks
        flows = np.load(os.path.join(folder_path, 'flows.npy'))
        masks = np.load(os.path.join(folder_path, 'masks.npy'))

        # Load ground truth if needed
        l_gt = np.load(os.path.join(folder_path, 'l_gt.npy'))

        # Convert to torch tensors
        I0 = torch.tensor(I0, dtype=torch.float32)
        I1 = torch.tensor(I1, dtype=torch.float32)
        flows = torch.tensor(flows, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)
        l_gt = torch.tensor(l_gt, dtype=torch.float32)

        # Apply transformations if any
        if self.transform:
            I0, I1, flows, masks, l_gt = self.transform(I0, I1, flows, masks, l_gt)

        return I0, I1, flows, masks, l_gt
