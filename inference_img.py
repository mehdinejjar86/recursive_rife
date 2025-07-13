import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab

def clear_write_buffer(user_args, matched_extension, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        path = os.path.join(user_args.output, f'{cnt:0>7d}{matched_extension}')
        cv2.imwrite(path, item[:, :, ::-1])
        cnt += 1

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)
            # convert grayscale to BGR
            if len(frame.shape) <  3:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = frame[:, :, ::-1].copy()
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n, model, scale=1.0):    
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
        return res
    else:
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2, model=model)
        second_half = make_inference(middle, I1, n=n//2, model=model)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]
        
def make_inference_recusrive(I0, I1, n, model, scale=1.0):    
    if model.version >= 3.9:
        res = []
        flows = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
        return res
    else:
      raise NotImplementedError("Recursive inference is not implemented for versions below 3.9")

def pad_image(img, padding):
        return F.pad(img, padding)


def main():
    
  warnings.filterwarnings("ignore")

  parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
  parser.add_argument('--model', dest='modelDir', type=str, default='rifev4.25', help='directory with trained model files')
  parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k images')
  parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k')
  parser.add_argument('--multi', dest='multi', type=int, default=2)
  parser.add_argument('--img', dest='img', type=str, default="Images path", help='input image directory')
  parser.add_argument('--output', dest='output', type=str, default='vid_out', help='output path')

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
  lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)
  lastframe_dtype=lastframe.dtype
  if lastframe_dtype == np.uint8:
      max_val=255.
  elif lastframe_dtype == np.uint16:
      max_val=65535.
  else:
      max_val=1.

  if len(lastframe.shape) < 3:
      lastframe = cv2.cvtColor(lastframe, cv2.COLOR_GRAY2BGR)
  lastframe = lastframe[:, :, ::-1].copy()
  videogen = videogen[1:]
  h, w, _ = lastframe.shape

  tmp = max(128, int(128 / args.scale))
  ph = ((h - 1) // tmp + 1) * tmp
  pw = ((w - 1) // tmp + 1) * tmp
  padding = (0, pw - w, 0, ph - h)
  pbar = tqdm(total=tot_frame)

  if args.output is None:
      args.output = 'vid_out'
  if not os.path.exists(args.output):
      os.makedirs(args.output)

  write_buffer = Queue(maxsize=500)
  read_buffer = Queue(maxsize=500)
  _thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
  _thread.start_new_thread(clear_write_buffer, (args, matched_extension, write_buffer))

  I1 = torch.from_numpy(np.transpose(lastframe.astype(np.int64), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / max_val
  I1 = pad_image(I1, padding=padding)
  temp = None # save lastframe when processing static frame

  while True:
      if temp is not None:
          frame = temp
          temp = None
      else:
          frame = read_buffer.get()
      if frame is None:
          break
      I0 = I1
      I1 = torch.from_numpy(np.transpose(frame.astype(np.int64), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / max_val
      I1 = pad_image(I1, padding=padding)
      I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
      I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
      ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

      break_flag = False
      if ssim > 0.996:
          frame = read_buffer.get() # read a new frame
          if frame is None:
              break_flag = True
              frame = lastframe
          else:
              temp = frame
          I1 = torch.from_numpy(np.transpose(frame.astype(np.int64), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / max_val
          I1 = pad_image(I1, padding=padding)
          I1 = model.inference(I0, I1, scale=args.scale)
          I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
          ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
          frame = (I1[0] * max_val).cpu().numpy().astype(lastframe_dtype).transpose(1, 2, 0)[:h, :w]
          
      if ssim < 0.2:
          output = []
          for i in range(args.multi - 1):
              output.append(I0)
          '''
          output = []
          step = 1 / args.multi
          alpha = 0
          for i in range(args.multi - 1):
              alpha += step
              beta = 1-alpha
              output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
          '''
      else:
          output = make_inference(I0, I1, args.multi - 1, model, scale=args.scale)

      write_buffer.put(lastframe)
      for mid in output:
          mid = (((mid[0] * max_val).cpu().numpy().astype(lastframe_dtype).transpose(1, 2, 0)))
          write_buffer.put(mid[:h, :w])
      pbar.update(1)
      lastframe = frame
      if break_flag:
          break


  write_buffer.put(lastframe)
  write_buffer.put(None)  

  import time
  while(not write_buffer.empty()):
      time.sleep(0.1)
  pbar.close()

if __name__ == '__main__':
    main()