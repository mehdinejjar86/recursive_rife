import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
from model.pytorch_msssim import ssim_matlab
from utility.imaging import read_image, save_image, pad_image

def main():

  warnings.filterwarnings("ignore")

  parser = argparse.ArgumentParser(
      description='Interpolation for a pair of images')
  parser.add_argument('--model', dest='modelDir', type=str,
                      default='ckpt/rifev4_25', help='directory with trained model files')
  parser.add_argument('--UHD', dest='UHD',
                      action='store_true', help='support 4k images')
  parser.add_argument('--scale', dest='scale', type=float,
                      default=1.0, help='Try scale=0.5 for 4k')
  parser.add_argument('--img', dest='img', type=str,
                      default="Images path", help='input image directory')
  parser.add_argument('--output', dest='output', type=str,
                      default='vid_out', help='output path')
  parser.add_argument(
    '--anchor',
    '-a',
    dest='anchor',
    type=int,            # <-- convert value to int
    default=3,           # pick whatever default you want
    help='Anchor frames'
  )

  args = parser.parse_args()

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  if args.UHD and args.scale == 1.0:
    args.scale = 0.5
  assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]

  if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
  else:
    device = torch.device("cpu")

  print(f"Using device: {device}")
  
  from ckpt.rifev4_25.RIFE_HDv3 import Model
  model = Model()
  if not hasattr(model, 'version'):
      model.version = 0
  model.load_model(args.modelDir, -1)
  print("Loaded 3.x/4.x HD model.")
  model.eval()
  model.device()

  videogen = []
  image_extensions = ('.png', '.tif', '.jpg', '.jpeg', '.bmp', '.gif')
  matched_extension = None

  for f in os.listdir(args.img):
    for ext in image_extensions:
      if f.lower().endswith(ext):
        matched_extension = ext
        break

    if matched_extension in f:
        videogen.append(f)

  if matched_extension:
    print(f"This is an image file with the '{matched_extension}' extension.")
  else:
    print("This is not an image file.")
    assert False, "The input directory does not contain valid image files."

  tot_frame = len(videogen)
  videogen.sort(key= lambda x:int(x[:-4]))
  # create a list with number of the frames
  ground_truth = [int(fname.split('.')[0]) for fname in videogen]
  
  print(f"Ground Truth frames: {ground_truth}")
  frame = read_image(os.path.join(args.img, videogen[0]), matched_extension)

  frame_dtype = frame.dtype
  if frame_dtype == np.uint8:
    max_val=255.
  elif frame_dtype == np.uint16:
    max_val=65535.
  else:
    max_val=1.

  print(f"Total frames: {tot_frame}, first frame shape: {frame.shape}, dtype: {frame_dtype}, max_val: {max_val}")
  
  h, w, _ = frame.shape

  tmp = max(128, int(128 / args.scale))
  ph = ((h - 1) // tmp + 1) * tmp
  pw = ((w - 1) // tmp + 1) * tmp
  padding = (0, pw - w, 0, ph - h)

  num_flows = args.anchor

  # Define exponential decay factor (you can adjust this value)
  decay_factor = 6.7  # You can experiment with different values

  # Calculate exponential weights
  indices = torch.arange(0, num_flows, device=device, dtype=torch.float32)
  flows_weights = torch.exp(-decay_factor * indices)

  # Reverse the weights so the most recent flows have the highest weight
  flows_weights = flows_weights.flip(0)

  # Normalize the weights so they sum to 1
  flows_weights = flows_weights / flows_weights.sum()

  print(f"Flows weights: {flows_weights}")
  
  for i in tqdm(range(len(videogen) - 1)):
    if i < args.anchor - 1 or i >= tot_frame - args.anchor:

      continue
    
    for frame in range(ground_truth[i] +1, ground_truth[i + 1]):
      flows = []
      masks = []
      for anchor in reversed(range(args.anchor)):
        I0_index = i - anchor
        I1_index = i + anchor + 1
        timestep = (frame - ground_truth[I0_index]) / (ground_truth[I1_index] - ground_truth[I0_index])
        
        I0 = read_image(os.path.join(args.img, videogen[I0_index]), matched_extension)
        I0 = torch.from_numpy(np.transpose(I0.astype(np.int64), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / max_val
        I0 = pad_image(I0, padding=padding)
        
        I1 = read_image(os.path.join(args.img, videogen[I1_index]), matched_extension)
        I1 = torch.from_numpy(np.transpose(I1.astype(np.int64), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / max_val
        I1 = pad_image(I1, padding=padding)
        
        flow, mask = model.flow_extractor(I0, I1, timestep, args.scale)
        
        flows.append(flow)
        masks.append(mask)
      
      output = model.inference_global(I0, I1, flows, masks, flows_weights, timestep, args.scale)
      save_image(output, args.output, frame, matched_extension, h, w, dtype=frame_dtype, max_val=max_val)

  
if __name__ == "__main__":
  main()