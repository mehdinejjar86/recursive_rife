import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
import math


def main():
    
  warnings.filterwarnings("ignore")

  parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
  parser.add_argument('--model', dest='modelDir', type=str, default='rifev4_25', help='directory with trained model files')
  parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k images')
  parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k')
  parser.add_argument('--multi', dest='multi', type=int, default=2)
  parser.add_argument('--img', dest='img', type=str, default="Images path", help='input image directory')
  parser.add_argument('--output', dest='output', type=str, default='vid_out', help='output path')
  parser.add_argument('--k', dest='k', action=int, help='Number of anchor points for interpolation')

  args = parser.parse_args()

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  