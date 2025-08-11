from torch.utils.data import DataLoader
from data_fusion import FlowMaskDataset

# Specify the root directory where your data folders are located
root_dir = 'dataset_fusion'

# Instantiate the dataset
dataset = FlowMaskDataset(root_dir)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Example usage during training
for I0, I1, flows, masks, l_gt in dataloader:
    print(flows)
    break
