import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import tifffile

def read_image(image_path, extension):
  flag_cv2 = False

  if extension == '.tif':
    image = tifffile.imread(image_path)
  else:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    flag_cv2 = True
  if len(image.shape) < 3:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

  if flag_cv2:
    image = image[:, :, ::-1].copy()  # Convert BGR to RGB
  return image

def save_image(tensor, output_path, name, extension, h, w, dtype=np.uint8, max_val=255.0):
    
  image = (tensor[0] * max_val).cpu().numpy().astype(dtype).transpose(1, 2, 0)[:h, :w]

  output_path = os.path.join(output_path, f"{name:0>7d}{extension}")

  if extension == '.tif':
      tifffile.imwrite(output_path, image[:, :, ::-1])
  else:
      cv2.imwrite(output_path, image[:, :, ::-1])  # Convert RGB back to BGR for saving

def make_inference(I0, I1, n, model, scale=1.0):    

  res = []
  for i in range(n):
      res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
  return res

def pad_image(img, padding):
        return F.pad(img, padding)

def main():
    
  warnings.filterwarnings("ignore")

  parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
  parser.add_argument('--model', dest='modelDir', type=str, default='rifev4_25', help='directory with trained model files')
  parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k images')
  parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k')
  parser.add_argument('--multi', dest='multi', type=int, default=2)
  parser.add_argument('--img', dest='img', type=str, default="Images path", help='input image directory')
  parser.add_argument('--output', dest='output', type=str, default='vid_out', help='output path')
  parser.add_argument('--recursive', dest='recursive', action='store_true', help='use recursive inference')

  args = parser.parse_args()

  if args.UHD and args.scale==1.0:
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

  torch.set_grad_enabled(False)

  from rifev4_25.RIFE_HDv3 import Model
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

  if matched_extension.lower() == '.tif':
    lastframe = tifffile.imread(os.path.join(args.img, videogen[0]))
  else:
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)

  lastframe_dtype = lastframe.dtype
  if lastframe_dtype == np.uint8:
    max_val=255.
  elif lastframe_dtype == np.uint16:
    max_val=65535.
  else:
    max_val=1.


  print(f"Total frames: {tot_frame}, first frame shape: {lastframe.shape}, dtype: {lastframe_dtype}, max_val: {max_val}")

  if len(lastframe.shape) < 3:
    lastframe = cv2.cvtColor(lastframe, cv2.COLOR_GRAY2BGR)

    lastframe = lastframe[:, :, ::-1].copy()

  h, w, _ = lastframe.shape

  tmp = max(128, int(128 / args.scale))
  ph = ((h - 1) // tmp + 1) * tmp
  pw = ((w - 1) // tmp + 1) * tmp
  padding = (0, pw - w, 0, ph - h)

  if args.output is None:
      args.output = 'vid_out'
  if not os.path.exists(args.output):
      os.makedirs(args.output)

  I1 = torch.from_numpy(np.transpose(lastframe.astype(np.int64), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / max_val
  I1 = pad_image(I1, padding=padding)

  name = 0
  save_image(I1, args.output, name, matched_extension, h, w, dtype=lastframe_dtype, max_val=max_val)

  name += 1

  for i in tqdm(range(len(videogen) - 1)):
      I0 = I1
      frame = read_image(os.path.join(args.img, videogen[i + 1]), matched_extension)
      I1 = torch.from_numpy(np.transpose(frame.astype(np.int64), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / max_val
      I1 = pad_image(I1, padding=padding)

      output = make_inference(I0, I1, args.multi - 1, model, scale=args.scale)

      for mid in output:
          save_image(mid, args.output, name, matched_extension, h, w, dtype=lastframe_dtype, max_val=max_val)
          name += 1

      save_image(I1, args.output, name, matched_extension, h, w, dtype=lastframe_dtype, max_val=max_val)
      name += 1

if __name__ == "__main__":
    main()
      