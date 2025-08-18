import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from datetime import datetime
import warnings
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

# Weights & Biases support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Import your dataset and model
from fusion_dataset import create_multi_dataloader
from fusion_model import FusionLoss, create_fusion_model


class ModelParallelWrapper(nn.Module):
    """
    Wrapper to enable model parallelism by splitting computation across GPUs
    This allows processing larger images by pooling GPU memory
    Updated for dynamic attention model
    """
    
    def __init__(self, base_model, max_attention_size, num_gpus=None, split_strategy='spatial'):
        super().__init__()
        
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.split_strategy = split_strategy
        self.max_attention_size = max_attention_size
        self.num_anchors = base_model.num_anchors
        self.base_channels = base_model.base_channels
        
        if self.num_gpus <= 1:
            # Single GPU - no parallelism needed
            self.model = base_model
            self.parallel_mode = False
        else:
            # Multiple GPUs - enable parallelism
            self.models = nn.ModuleList()
            for i in range(self.num_gpus):
                # Create a model copy on each GPU with dynamic attention
                model_copy = create_fusion_model(
                    num_anchors=self.num_anchors,
                    base_channels=self.base_channels,
                    max_attention_size=max_attention_size
                )
                # Move entire model to the specific GPU
                model_copy = model_copy.cuda(i)
                
                # Ensure all submodules are on the correct device
                for module in model_copy.modules():
                    if hasattr(module, 'to'):
                        module.to(f'cuda:{i}')
                
                self.models.append(model_copy)
            self.parallel_mode = True
            
            # Share parameters across GPUs (they process different parts of same batch)
            if self.num_gpus > 1:
                # Sync parameters from GPU 0 to all others
                source_state = self.models[0].state_dict()
                for i in range(1, self.num_gpus):
                    # Load the state dict to ensure proper parameter sharing
                    self.models[i].load_state_dict(source_state)
            
    def sync_gradients(self):
        """Synchronize gradients across model copies when using model parallelism"""
        if not self.parallel_mode or self.num_gpus <= 1:
            return
        
        # Average gradients from all GPUs to GPU 0
        for param_idx, param_main in enumerate(self.models[0].parameters()):
            if param_main.grad is None:
                continue
                
            # Collect gradients from all GPUs
            grads = [param_main.grad.data]
            for i in range(1, self.num_gpus):
                param_copy = list(self.models[i].parameters())[param_idx]
                if param_copy.grad is not None:
                    grads.append(param_copy.grad.data.cuda(0))
            
            # Average gradients
            if len(grads) > 1:
                avg_grad = torch.stack(grads).mean(dim=0)
                param_main.grad.data = avg_grad
    
    def sync_parameters(self):
        """Synchronize parameters from GPU 0 to all other GPUs after optimizer step"""
        if not self.parallel_mode or self.num_gpus <= 1:
            return
        
        # Copy parameters from GPU 0 to all other GPUs
        source_state = self.models[0].state_dict()
        for i in range(1, self.num_gpus):
            # Create a state dict with tensors on the target device
            target_state = {}
            for key, value in source_state.items():
                if torch.is_tensor(value):
                    target_state[key] = value.cuda(i)
                else:
                    target_state[key] = value
            self.models[i].load_state_dict(target_state)
    
    def forward(self, I0_all, I1_all, flows_all, masks_all, timesteps):
        """Forward pass with optional model parallelism"""
        
        if not self.parallel_mode:
            # Single GPU path
            return self.model(I0_all, I1_all, flows_all, masks_all, timesteps)
        
        B, A, C, H, W = I0_all.shape
        
        if self.split_strategy == 'spatial':
            # Split images spatially across GPUs
            return self._forward_spatial_parallel(
                I0_all, I1_all, flows_all, masks_all, timesteps
            )
        elif self.split_strategy == 'anchor':
            # Split anchors across GPUs
            return self._forward_anchor_parallel(
                I0_all, I1_all, flows_all, masks_all, timesteps
            )
        else:
            # Default to batch splitting
            return self._forward_batch_parallel(
                I0_all, I1_all, flows_all, masks_all, timesteps
            )
    
    def _forward_spatial_parallel(self, I0_all, I1_all, flows_all, masks_all, timesteps):
        """Split images spatially (by height) across GPUs"""
        B, A, C, H, W = I0_all.shape
        
        # Calculate split size with overlap
        overlap = 32
        base_strip_height = H // self.num_gpus
        
        outputs = []
        aux_outputs_list = []
        strip_infos = []  # Store strip information for merging
        
        for gpu_id in range(self.num_gpus):
            # Calculate strip boundaries with overlap
            start_h = max(0, gpu_id * base_strip_height - overlap // 2)
            end_h = min(H, (gpu_id + 1) * base_strip_height + overlap // 2)
            
            # Store strip info
            strip_infos.append({
                'start_h': start_h,
                'end_h': end_h,
                'gpu_id': gpu_id
            })
            
            # Extract strip and move to appropriate GPU
            with torch.cuda.device(gpu_id):
                I0_strip = I0_all[:, :, :, start_h:end_h, :].cuda(gpu_id)
                I1_strip = I1_all[:, :, :, start_h:end_h, :].cuda(gpu_id)
                flows_strip = flows_all[:, :, :, start_h:end_h, :].cuda(gpu_id)
                masks_strip = masks_all[:, :, start_h:end_h, :].cuda(gpu_id)
                timesteps_gpu = timesteps.cuda(gpu_id)
                
                # Process strip on its GPU
                output_strip, aux_strip = self.models[gpu_id](
                    I0_strip, I1_strip, flows_strip, masks_strip, timesteps_gpu
                )
            
            outputs.append(output_strip)
            aux_outputs_list.append(aux_strip)
        
        # Merge outputs (blend overlapping regions)
        merged_output = self._merge_spatial_outputs(outputs, H, overlap, base_strip_height)
        
        # Merge auxiliary outputs properly
        aux_outputs = self._merge_aux_outputs_spatial(aux_outputs_list, strip_infos, H, W, overlap, base_strip_height)
        
        return merged_output, aux_outputs
    
    def _merge_aux_outputs_spatial(self, aux_outputs_list, strip_infos, full_H, full_W, overlap, base_strip_height):
        """Merge auxiliary outputs from spatial parallelism"""
        if not aux_outputs_list:
            return {}
        
        merged_aux = {}
        
        for key in aux_outputs_list[0].keys():
            if key == 'warped_imgs':
                # Special handling for warped_imgs - need to merge spatial strips
                warped_list = []
                for warped_idx in range(len(aux_outputs_list[0]['warped_imgs'])):
                    # Collect this warped image from all GPUs
                    strips = []
                    for gpu_idx, aux in enumerate(aux_outputs_list):
                        strip = aux['warped_imgs'][warped_idx].cuda(0)  # Move to GPU 0
                        strips.append(strip)
                    
                    # Merge the strips for this warped image
                    merged_warped = self._merge_spatial_tensors(strips, full_H, overlap, base_strip_height)
                    warped_list.append(merged_warped)
                
                merged_aux['warped_imgs'] = warped_list
                
            elif key == 'refined_masks':
                # Similar handling for refined_masks
                mask_list = []
                for mask_idx in range(len(aux_outputs_list[0]['refined_masks'])):
                    strips = []
                    for gpu_idx, aux in enumerate(aux_outputs_list):
                        strip = aux['refined_masks'][mask_idx].cuda(0)
                        strips.append(strip)
                    
                    merged_mask = self._merge_spatial_tensors(strips, full_H, overlap, base_strip_height)
                    mask_list.append(merged_mask)
                
                merged_aux['refined_masks'] = mask_list
                
            elif isinstance(aux_outputs_list[0][key], list):
                # For other lists, just use the first GPU's version
                merged_aux[key] = aux_outputs_list[0][key]
                
            elif torch.is_tensor(aux_outputs_list[0][key]):
                # For single tensors, check if they need spatial merging
                if aux_outputs_list[0][key].dim() >= 3 and aux_outputs_list[0][key].shape[-2] < full_H:
                    # This tensor has spatial dimensions that need merging
                    strips = [aux[key].cuda(0) for aux in aux_outputs_list]
                    merged_aux[key] = self._merge_spatial_tensors(strips, full_H, overlap, base_strip_height)
                else:
                    # This tensor doesn't have spatial dimensions or is already full size
                    merged_aux[key] = aux_outputs_list[0][key].cuda(0)
            else:
                # Non-tensor values
                merged_aux[key] = aux_outputs_list[0][key]
        
        return merged_aux
    
    def _merge_spatial_tensors(self, strips, full_height, overlap, base_strip_height):
        """Merge spatial tensor strips with blending"""
        if not strips:
            return None
        
        device = strips[0].device
        shape = list(strips[0].shape)
        
        # Determine the dimension structure
        if len(shape) == 4:  # [B, C, H, W]
            shape[-2] = full_height  # Update height dimension
            H_dim = -2
            W_dim = -1
        elif len(shape) == 3:  # [B, H, W] or [C, H, W]
            if shape[-2] < shape[-1]:  # Likely [B/C, H, W]
                shape[-2] = full_height
                H_dim = -2
                W_dim = -1
            else:
                # Handle other possible arrangements
                shape[-2] = full_height
                H_dim = -2
                W_dim = -1
        else:
            # For other shapes, assume second-to-last is height
            shape[-2] = full_height
            H_dim = -2
            W_dim = -1
        
        # Initialize merged tensor and weights
        merged = torch.zeros(shape, device=device)
        
        # Create weight tensor with same dimensions as strips but single channel for weighting
        if len(shape) == 4:  # [B, C, H, W]
            weight_shape = [shape[0], 1, full_height, shape[3]]
        elif len(shape) == 3:  # [B, H, W]
            weight_shape = [shape[0], full_height, shape[2]]
        else:
            weight_shape = shape.copy()
        
        weights = torch.zeros(weight_shape, device=device)
        
        for gpu_id, strip in enumerate(strips):
            start_h = max(0, gpu_id * base_strip_height - overlap // 2)
            end_h = min(full_height, (gpu_id + 1) * base_strip_height + overlap // 2)
            strip_h = strip.shape[H_dim]
            
            # Create weight for blending with proper shape
            if len(strip.shape) == 4:  # [B, C, H, W]
                B, C, H_s, W = strip.shape
                weight = torch.ones(B, 1, H_s, W, device=device)
                
                if overlap > 0:
                    # Fade in at top (except for first strip)
                    if gpu_id > 0 and overlap // 2 < strip_h:
                        fade_size = min(overlap // 2, strip_h)
                        fade = torch.linspace(0, 1, fade_size, device=device)
                        # Reshape fade for broadcasting
                        fade = fade.view(1, 1, -1, 1)
                        weight[:, :, :fade_size, :] *= fade
                    
                    # Fade out at bottom (except for last strip)
                    if gpu_id < len(strips) - 1 and overlap // 2 < strip_h:
                        fade_size = min(overlap // 2, strip_h)
                        fade = torch.linspace(1, 0, fade_size, device=device)
                        # Reshape fade for broadcasting
                        fade = fade.view(1, 1, -1, 1)
                        weight[:, :, -fade_size:, :] *= fade
                
                # Apply weighted strip
                weighted_strip = strip * weight.expand(-1, C, -1, -1)
                merged[:, :, start_h:end_h, :] += weighted_strip
                weights[:, :, start_h:end_h, :] += weight
                
            elif len(strip.shape) == 3:  # [B, H, W]
                B, H_s, W = strip.shape
                weight = torch.ones(B, H_s, 1, device=device)
                
                if overlap > 0:
                    # Fade in at top
                    if gpu_id > 0 and overlap // 2 < strip_h:
                        fade_size = min(overlap // 2, strip_h)
                        fade = torch.linspace(0, 1, fade_size, device=device)
                        fade = fade.view(1, -1, 1)
                        weight[:, :fade_size, :] *= fade
                    
                    # Fade out at bottom
                    if gpu_id < len(strips) - 1 and overlap // 2 < strip_h:
                        fade_size = min(overlap // 2, strip_h)
                        fade = torch.linspace(1, 0, fade_size, device=device)
                        fade = fade.view(1, -1, 1)
                        weight[:, -fade_size:, :] *= fade
                
                # Apply weighted strip
                weighted_strip = strip * weight.expand(-1, -1, W)
                merged[:, start_h:end_h, :] += weighted_strip
                weights[:, start_h:end_h, :] += weight.expand(-1, -1, W)
                
            else:
                # Fallback for other tensor shapes - no blending
                if H_dim == -2:
                    merged[..., start_h:end_h, :] += strip
                    if len(weight_shape) == len(strip.shape):
                        weights[..., start_h:end_h, :] += 1.0
                else:
                    # Handle unexpected dimension arrangement
                    merged[..., start_h:end_h] += strip
                    if len(weight_shape) == len(strip.shape):
                        weights[..., start_h:end_h] += 1.0
        
        # Normalize by weights
        if len(shape) == 4:
            # Expand weights to match channel dimension
            weights_expanded = weights.expand(-1, shape[1], -1, -1)
            merged = merged / (weights_expanded + 1e-8)
        elif len(shape) == 3:
            merged = merged / (weights + 1e-8)
        else:
            merged = merged / (weights + 1e-8)
        
        return merged
    
    def _forward_anchor_parallel(self, I0_all, I1_all, flows_all, masks_all, timesteps):
        """Split anchors across GPUs"""
        B, A, C, H, W = I0_all.shape
        anchors_per_gpu = A // self.num_gpus + (1 if A % self.num_gpus else 0)
        
        outputs = []
        aux_outputs_list = []
        
        for gpu_id in range(self.num_gpus):
            start_a = gpu_id * anchors_per_gpu
            end_a = min(start_a + anchors_per_gpu, A)
            
            if start_a >= A:
                break
            
            # Extract anchors for this GPU
            with torch.cuda.device(gpu_id):
                I0_gpu = I0_all[:, start_a:end_a].cuda(gpu_id)
                I1_gpu = I1_all[:, start_a:end_a].cuda(gpu_id)
                flows_gpu = flows_all[:, start_a:end_a].cuda(gpu_id)
                masks_gpu = masks_all[:, start_a:end_a].cuda(gpu_id)
                timesteps_gpu = timesteps[:, start_a:end_a].cuda(gpu_id)
                
                # Process on GPU
                output_gpu, aux_gpu = self.models[gpu_id](
                    I0_gpu, I1_gpu, flows_gpu, masks_gpu, timesteps_gpu
                )
            
            outputs.append(output_gpu.cuda(0))  # Move to GPU 0 for merging
            aux_outputs_list.append(aux_gpu)
        
        # Average outputs from different anchors
        merged_output = torch.stack(outputs, dim=0).mean(dim=0)
        
        # Merge auxiliary outputs
        aux_outputs = {}
        for key in aux_outputs_list[0].keys():
            if isinstance(aux_outputs_list[0][key], list):
                aux_outputs[key] = aux_outputs_list[0][key]
            else:
                aux_outputs[key] = aux_outputs_list[0][key].cuda(0) if torch.is_tensor(aux_outputs_list[0][key]) else aux_outputs_list[0][key]
        
        return merged_output, aux_outputs
    
    def _forward_batch_parallel(self, I0_all, I1_all, flows_all, masks_all, timesteps):
        """Split batch across GPUs"""
        B = I0_all.shape[0]
        samples_per_gpu = B // self.num_gpus + (1 if B % self.num_gpus else 0)
        
        outputs = []
        aux_outputs_list = []
        
        for gpu_id in range(self.num_gpus):
            start_b = gpu_id * samples_per_gpu
            end_b = min(start_b + samples_per_gpu, B)
            
            if start_b >= B:
                break
            
            # Extract batch for this GPU
            with torch.cuda.device(gpu_id):
                I0_gpu = I0_all[start_b:end_b].cuda(gpu_id)
                I1_gpu = I1_all[start_b:end_b].cuda(gpu_id)
                flows_gpu = flows_all[start_b:end_b].cuda(gpu_id)
                masks_gpu = masks_all[start_b:end_b].cuda(gpu_id)
                timesteps_gpu = timesteps[start_b:end_b].cuda(gpu_id)
                
                # Process on GPU
                output_gpu, aux_gpu = self.models[gpu_id](
                    I0_gpu, I1_gpu, flows_gpu, masks_gpu, timesteps_gpu
                )
            
            outputs.append(output_gpu.cuda(0))  # Move to GPU 0 for concatenation
            aux_outputs_list.append(aux_gpu)
        
        # Concatenate outputs
        merged_output = torch.cat(outputs, dim=0)
        
        # Merge auxiliary outputs
        aux_outputs = {}
        for key in aux_outputs_list[0].keys():
            if isinstance(aux_outputs_list[0][key], list):
                aux_outputs[key] = aux_outputs_list[0][key]
            else:
                aux_values = [aux[key].cuda(0) if torch.is_tensor(aux[key]) else aux[key] 
                              for aux in aux_outputs_list]
                if torch.is_tensor(aux_values[0]):
                    aux_outputs[key] = torch.cat(aux_values, dim=0)
                else:
                    aux_outputs[key] = aux_values[0]
        
        return merged_output, aux_outputs
    
    def _merge_spatial_outputs(self, outputs, full_height, overlap, base_strip_height):
        """Merge spatially split outputs with blending"""
        device = outputs[0].device
        B, C, _, W = outputs[0].shape
        
        # Initialize full output
        merged = torch.zeros(B, C, full_height, W, device=device)
        weights = torch.zeros(B, 1, full_height, W, device=device)
        
        for gpu_id, output in enumerate(outputs):
            start_h = max(0, gpu_id * base_strip_height - overlap // 2)
            end_h = min(full_height, (gpu_id + 1) * base_strip_height + overlap // 2)
            strip_h = end_h - start_h
            
            # Create weight for blending (fade at edges)
            weight = torch.ones(B, 1, strip_h, W, device=device)
            
            if overlap > 0:
                # Fade in at top (except for first strip)
                if gpu_id > 0 and overlap // 2 < strip_h:
                    fade_size = min(overlap // 2, strip_h)
                    fade = torch.linspace(0, 1, fade_size, device=device)
                    weight[:, :, :fade_size, :] *= fade.view(1, 1, -1, 1)
                
                # Fade out at bottom (except for last strip)
                if gpu_id < self.num_gpus - 1 and overlap // 2 < strip_h:
                    fade_size = min(overlap // 2, strip_h)
                    fade = torch.linspace(1, 0, fade_size, device=device)
                    weight[:, :, -fade_size:, :] *= fade.view(1, 1, -1, 1)
            
            # Move output to GPU 0 for merging
            output = output.cuda(0)
            weight = weight.cuda(0)
            
            # Add weighted output
            merged[:, :, start_h:end_h, :] += output * weight
            weights[:, :, start_h:end_h, :] += weight
        
        # Normalize by weights
        merged = merged / (weights + 1e-8)
        
        return merged


class MetricsCalculator:
    """Calculate PSNR and SSIM metrics at specified resolution"""
    
    def __init__(self, eval_size=(256, 256)):
        self.eval_size = eval_size
    
    def calculate_psnr(self, pred, target):
        """Calculate PSNR between predicted and target images"""
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        if pred.ndim == 4:
            pred = pred.transpose(0, 2, 3, 1)
            target = target.transpose(0, 2, 3, 1)
        elif pred.ndim == 3 and pred.shape[0] == 3:
            pred = pred.transpose(1, 2, 0)
            target = target.transpose(1, 2, 0)
        
        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)
        
        if pred.ndim == 4:
            psnr_values = []
            for i in range(pred.shape[0]):
                psnr_val = psnr_func(target[i], pred[i], data_range=1.0)
                psnr_values.append(psnr_val)
            return np.mean(psnr_values)
        else:
            return psnr_func(target, pred, data_range=1.0)
    
    def calculate_ssim(self, pred, target):
        """Calculate SSIM between predicted and target images"""
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        if pred.ndim == 4:
            pred = pred.transpose(0, 2, 3, 1)
            target = target.transpose(0, 2, 3, 1)
        elif pred.ndim == 3 and pred.shape[0] == 3:
            pred = pred.transpose(1, 2, 0)
            target = target.transpose(1, 2, 0)
        
        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)
        
        if pred.ndim == 4:
            ssim_values = []
            for i in range(pred.shape[0]):
                ssim_val = ssim_func(target[i], pred[i], 
                                    data_range=1.0, 
                                    channel_axis=2 if pred.shape[-1] == 3 else None)
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        else:
            return ssim_func(target, pred, 
                           data_range=1.0,
                           channel_axis=2 if pred.shape[-1] == 3 else None)
    
    def resize_for_metrics(self, tensor, size=None):
        """Resize tensor to specified size for metric calculation"""
        if size is None:
            size = self.eval_size
        
        if tensor.shape[-2:] == size:
            return tensor
        
        resized = torch.nn.functional.interpolate(
            tensor if tensor.dim() == 4 else tensor.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False
        )
        
        return resized if tensor.dim() == 4 else resized.squeeze(0)


class ParallelFusionTrainer:
    """Trainer with Model Parallelism for Large Images and W&B Support"""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        
        # Create directories
        self.run_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.run_name = f"{config.run_name}_{self.run_timestamp}" if hasattr(config, 'run_name') and config.run_name else self.run_timestamp
        
        self.config.checkpoint_dir = Path("runs") / self.run_name / Path(self.config.checkpoint_dir)
        self.config.log_dir = Path("runs") / self.run_name / Path(self.config.log_dir)
        self.config.sample_dir = Path("runs") / self.run_name / Path(self.config.sample_dir)
        
        self.setup_directories()
        
        # Initialize W&B if available and requested
        self.use_wandb = WANDB_AVAILABLE and getattr(config, 'use_wandb', False)
        if self.use_wandb:
            self._init_wandb()
        
        # Create dataloader
        self.train_loader, self.train_dataset = self._create_dataloader()
        
        # Create model with parallelism if multiple GPUs available
        self.model = self._create_parallel_model()
        
        # Create loss function
        self.loss_fn = FusionLoss(
            lambda_l1=config.lambda_l1,
            lambda_perceptual=config.lambda_perceptual,
            lambda_smooth=config.lambda_smooth,
            lambda_consistency=config.lambda_consistency
        ).to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup metrics
        metric_size = tuple(config.metric_size) if hasattr(config, 'metric_size') else (256, 256)
        self.metrics_calc = MetricsCalculator(eval_size=metric_size)
        
        # Setup logging
        self.writer = SummaryWriter(self.config.log_dir)
        self.start_epoch = 0
        self.best_psnr = 0
        self.best_ssim = 0
        self.epoch_metrics = []
        
        # Load checkpoint if exists
        if config.resume:
            self.load_checkpoint(config.resume)
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb_config = vars(self.config).copy()
        wandb_config['run_timestamp'] = self.run_timestamp
        wandb_config['num_gpus'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Get max attention size info
        if hasattr(self.config, 'max_attention_size'):
            size = int(np.sqrt(self.config.max_attention_size))
            wandb_config['max_attention_dim'] = f"{size}×{size}"
        
        wandb.init(
            project=getattr(self.config, 'wandb_project', 'fusion-model'),
            entity=getattr(self.config, 'wandb_entity', None),
            name=self.run_name,
            config=wandb_config,
            resume=self.config.resume is not None
        )
        
        # Watch model for gradient tracking
        if getattr(self.config, 'wandb_watch_model', False):
            if hasattr(self.model, 'models'):
                wandb.watch(self.model.models[0], log='all', log_freq=100)
            else:
                wandb.watch(self.model, log='all', log_freq=100)
        
        print(f"W&B initialized: {wandb.run.url}")
    
    def _setup_device(self):
        """Setup computing device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            num_gpus = torch.cuda.device_count()
            print(f"Using {num_gpus} CUDA GPU(s)")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    
    def setup_directories(self):
        """Create directory structure"""
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.log_dir = Path(self.config.log_dir)
        self.sample_dir = Path(self.config.sample_dir)
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nRun directory: {self.run_name}")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Logs: {self.log_dir}")
        print(f"  Samples: {self.sample_dir}")
        
        # Save config
        config_dict = vars(self.config)
        config_dict['run_timestamp'] = self.run_timestamp
        config_dict['run_name'] = self.run_name
        config_dict['num_gpus'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)


    def _create_dataloader(self):
        """Create training dataloader"""
        gt_paths = self.config.gt_paths
        steps = self.config.steps
        
        if isinstance(steps, int):
            steps = [steps] * len(gt_paths)
        elif len(steps) == 1 and len(gt_paths) > 1:
            steps = steps * len(gt_paths)
        
        mix_strategy = getattr(self.config, 'mix_strategy', 'uniform')
        path_weights = getattr(self.config, 'path_weights', None)
        cache_flows = getattr(self.config, 'cache_flows', False)
        
        # Adjust num_workers for MPS or when using model parallelism
        num_workers = self.config.num_workers
        if self.device.type == 'mps' and num_workers > 0:
            print("Warning: MPS device detected. Setting num_workers=0")
            num_workers = 0
        
        # When using model parallelism with multiple GPUs, reduce workers to avoid device conflicts
        use_parallel = getattr(self.config, 'model_parallel', False)
        if use_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            if num_workers > 0:
                print(f"Warning: Model parallelism detected. Reducing num_workers from {num_workers} to 0 to avoid device conflicts")
                num_workers = 0
        
        # REMOVE pin_memory argument - DataLoader will handle it internally
        train_dataloader, train_dataset = create_multi_dataloader(
            gt_paths=gt_paths,
            steps=steps,
            anchor=self.config.num_anchors,
            scale=self.config.scale,
            UHD=self.config.UHD,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            # pin_memory=pin_memory,  # ← REMOVE THIS LINE
            model_dir=self.config.rife_model_dir,
            mix_strategy=mix_strategy,
            path_weights=path_weights,
            cache_flows=cache_flows
        )
        
        print("\nDataset Configuration:")
        print(f"Total samples: {len(train_dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Batches per epoch: {len(train_dataloader)}")
        print(f"Num workers: {num_workers}")
        
        return train_dataloader, train_dataset
    
    def _create_parallel_model(self):
        """Create model with optional parallelism"""
        # Get max attention size for dynamic attention
        max_attention_size = getattr(self.config, 'max_attention_size', 96*96)
        
        base_model = create_fusion_model(
            num_anchors=self.config.num_anchors,
            base_channels=self.config.base_channels,
            max_attention_size=max_attention_size
        )
        
        # Print attention configuration
        size = int(np.sqrt(max_attention_size))
        print(f"\nDynamic Hierarchical Attention Configuration:")
        print(f"  Max attention size: {size}×{size} = {max_attention_size:,} elements")
        print(f"  Memory usage estimate: ~{max_attention_size * 4 * 4 / 1e9:.2f} GB per head")
        
        # Check if we should use model parallelism
        use_parallel = getattr(self.config, 'model_parallel', False)
        parallel_strategy = getattr(self.config, 'parallel_strategy', 'spatial')
        
        if use_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Enabling Model Parallelism with {parallel_strategy} strategy")
            model = ModelParallelWrapper(
                base_model,
                max_attention_size=max_attention_size,
                num_gpus=torch.cuda.device_count(),
                split_strategy=parallel_strategy
            )
        else:
            model = base_model.to(self.device)
            print(f"Using standard model on {self.device}")
        
        total_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
        
        return model
    
    def _create_optimizer(self):
        """Create optimizer"""
        if hasattr(self.model, 'models'):
            # Model parallel - optimize only the first model (others are synchronized)
            # We'll manually sync gradients across GPUs
            params = self.model.models[0].parameters()
        else:
            params = self.model.parameters()
        
        if self.config.optimizer == 'adam':
            optimizer = optim.Adam(
                params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        else:
            return None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {'total': 0, 'l1': 0, 'perceptual': 0, 'consistency': 0}
        epoch_psnr = 0
        epoch_ssim = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            I0_all = batch['I0'].to(self.device, non_blocking=True)
            I1_all = batch['I1'].to(self.device, non_blocking=True)
            flows_all = batch['flows'].to(self.device, non_blocking=True)
            masks_all = batch['masks'].to(self.device, non_blocking=True)
            timesteps = batch['timesteps'].to(self.device, non_blocking=True)
            I_gt = batch['I_gt'].to(self.device, non_blocking=True)
            
            # Forward pass
            output, aux_outputs = self.model(I0_all, I1_all, flows_all, masks_all, timesteps)
            
            # Ensure output is on same device as ground truth
            if output.device != I_gt.device:
                output = output.to(I_gt.device)
            
            # Compute loss
            losses = self.loss_fn(output, I_gt, aux_outputs)
            total_loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Synchronize gradients if using model parallelism
            if hasattr(self.model, 'sync_gradients'):
                self.model.sync_gradients()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                if hasattr(self.model, 'models'):
                    # Only clip gradients on the main model (GPU 0)
                    nn.utils.clip_grad_norm_(self.model.models[0].parameters(), self.config.grad_clip)
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            # Synchronize parameters if using model parallelism
            if hasattr(self.model, 'sync_parameters'):
                self.model.sync_parameters()
            
            # Calculate metrics
            with torch.no_grad():
                output_resized = self.metrics_calc.resize_for_metrics(output)
                I_gt_resized = self.metrics_calc.resize_for_metrics(I_gt)
                
                batch_psnr = self.metrics_calc.calculate_psnr(output_resized, I_gt_resized)
                batch_ssim = self.metrics_calc.calculate_ssim(output_resized, I_gt_resized)
                
                epoch_psnr += batch_psnr
                epoch_ssim += batch_ssim
            
            # Update epoch losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'psnr': f"{batch_psnr:.2f}",
                'ssim': f"{batch_ssim:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to W&B periodically
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'batch/loss': total_loss.item(),
                    'batch/psnr': batch_psnr,
                    'batch/ssim': batch_ssim,
                    'batch/lr': self.optimizer.param_groups[0]['lr'],
                    'step': epoch * len(self.train_loader) + batch_idx
                })
            
            # Report GPU memory usage periodically
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_mb = torch.cuda.memory_allocated(i) / 1024**2
                    max_mem_mb = torch.cuda.max_memory_allocated(i) / 1024**2
                    if mem_mb > 0:
                        pbar.set_description(
                            f"Epoch {epoch}/{self.config.epochs} | GPU{i}: {mem_mb:.0f}/{max_mem_mb:.0f}MB"
                        )
                        if self.use_wandb:
                            wandb.log({
                                f'gpu_{i}/memory_mb': mem_mb,
                                f'gpu_{i}/max_memory_mb': max_mem_mb,
                                'step': epoch * len(self.train_loader) + batch_idx
                            })
        
        # Average metrics
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        epoch_psnr /= num_batches
        epoch_ssim /= num_batches
        
        epoch_losses['psnr'] = epoch_psnr
        epoch_losses['ssim'] = epoch_ssim
        
        return epoch_losses
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        if hasattr(self.model, 'models'):
            # Save first model's state (they're synchronized)
            model_state = self.model.models[0].state_dict()
            # Move all tensors to CPU for saving
            cpu_state = {}
            for key, value in model_state.items():
                if torch.is_tensor(value):
                    cpu_state[key] = value.cpu()
                else:
                    cpu_state[key] = value
            model_state = cpu_state
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'epoch_metrics': self.epoch_metrics,
            'config': vars(self.config),
            'wandb_run_id': wandb.run.id if self.use_wandb else None
        }
        
        if epoch % self.config.save_interval == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
            
            if self.use_wandb:
                wandb.save(str(checkpoint_path))
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  Best model saved!")
            
            if self.use_wandb:
                wandb.save(str(best_path))
                wandb.run.summary["best_psnr"] = self.best_psnr
                wandb.run.summary["best_ssim"] = self.best_ssim
        
        latest_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Load to CPU first
        
        if hasattr(self.model, 'models'):
            # Load to all model copies with proper device placement
            state_dict = checkpoint['model_state_dict']
            for i, model in enumerate(self.model.models):
                # Create state dict with tensors on the correct device
                device_state = {}
                for key, value in state_dict.items():
                    if torch.is_tensor(value):
                        device_state[key] = value.cuda(i)
                    else:
                        device_state[key] = value
                model.load_state_dict(device_state)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_psnr = checkpoint.get('best_psnr', 0)
        self.best_ssim = checkpoint.get('best_ssim', 0)
        self.epoch_metrics = checkpoint.get('epoch_metrics', [])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.start_epoch}")
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config.epochs} epochs")
        print("-" * 60)
        
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start_time = datetime.now()
            
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Check if best model
            is_best = train_losses['psnr'] > self.best_psnr
            if is_best:
                self.best_psnr = train_losses['psnr']
                self.best_ssim = train_losses['ssim']
                print(f"  New best model! PSNR: {self.best_psnr:.2f} dB")
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics to TensorBoard
            for name, value in train_losses.items():
                self.writer.add_scalar(f'train/{name}', value, epoch)
            
            self.writer.add_scalar('learning_rate', 
                                  self.optimizer.param_groups[0]['lr'], epoch)
            
            # Log metrics to W&B
            if self.use_wandb:
                wandb_metrics = {f'epoch/{k}': v for k, v in train_losses.items()}
                wandb_metrics['epoch'] = epoch
                wandb_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
                wandb_metrics['is_best'] = is_best
                wandb.log(wandb_metrics)
            
            # Print summary
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            print(f"\nEpoch {epoch}/{self.config.epochs} ({epoch_time:.1f}s)")
            print(f"  Loss: {train_losses['total']:.4f}")
            print(f"  PSNR: {train_losses['psnr']:.2f} dB")
            print(f"  SSIM: {train_losses['ssim']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Store metrics
            self.epoch_metrics.append({
                'epoch': epoch,
                'train': train_losses,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Save metrics JSON
            with open(self.checkpoint_dir / 'metrics.json', 'w') as f:
                json.dump(self.epoch_metrics, f, indent=2, default=str)
        
        self.writer.close()
        
        if self.use_wandb:
            wandb.finish()
        
        print("\nTraining completed!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Best SSIM: {self.best_ssim:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Fusion Model with Dynamic Attention and W&B Support')
    
    # Dataset arguments
    parser.add_argument('--gt_paths', '--gt_path', type=str, nargs='+', required=True,
                        dest='gt_paths',
                        help='Path(s) to ground truth images')
    parser.add_argument('--steps', '--step', type=int, nargs='+', required=True,
                        dest='steps',
                        help='Step size(s) for sequence generation')
    
    # Validation data arguments
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (0.0 to disable, default: 0.1)')
    parser.add_argument('--val_paths', '--val_path', type=str, nargs='+', default=None,
                        dest='val_paths',
                        help='Separate validation dataset paths (optional)')
    parser.add_argument('--val_steps', '--val_step', type=int, nargs='+', default=None,
                        dest='val_steps',
                        help='Step sizes for validation paths')
    
    # Model arguments
    parser.add_argument('--num_anchors', type=int, default=3,
                        help='Number of anchor frames')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels in model')
    parser.add_argument('--max_attention_size', type=int, default=96*96,
                        help='Maximum size for attention matrices (e.g., 64*64, 96*96, 128*128)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for processing')
    parser.add_argument('--UHD', action='store_true',
                        help='Support for 4K images')
    parser.add_argument('--rife_model_dir', type=str, default='ckpt/rifev4_25',
                        help='Path to RIFE model directory')
    
    # Parallelism arguments
    parser.add_argument('--model_parallel', action='store_true',
                        help='Enable model parallelism for large images')
    parser.add_argument('--parallel_strategy', type=str, default='spatial',
                        choices=['spatial', 'anchor', 'batch'],
                        help='Parallelism strategy: spatial (split image), anchor (split anchors), batch')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    
    # Loss weights
    parser.add_argument('--lambda_l1', type=float, default=1.0,
                        help='Weight for L1 loss')
    parser.add_argument('--lambda_perceptual', type=float, default=0.1,
                        help='Weight for perceptual loss')
    parser.add_argument('--lambda_smooth', type=float, default=0.01,
                        help='Weight for smoothness loss')
    parser.add_argument('--lambda_consistency', type=float, default=0.1,
                        help='Weight for consistency loss')
    
    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    
    # Logging and saving
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory for sample predictions')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional name for this training run')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint saving interval')
    
    # W&B arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='fusion-model',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity (username or team)')
    parser.add_argument('--wandb_watch_model', action='store_true',
                        help='Watch model gradients in W&B')
    
    # Metrics
    parser.add_argument('--metric_size', type=int, nargs=2, default=[256, 256],
                        help='Size for computing metrics')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--force_single_worker', action='store_true',
                        help='Force single worker data loading (useful for debugging)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--mix_strategy', type=str, default='uniform',
                        help='Dataset mixing strategy')
    parser.add_argument('--cache_flows', action='store_true',
                        help='Cache extracted flows (recommended when using model parallelism)')
    parser.add_argument('--precache_flows', action='store_true',
                        help='Pre-cache all flows before training starts (uses more memory but avoids runtime extraction)')
    
    args = parser.parse_args()
    
    # Force single worker if requested or if using model parallelism without flow caching
    if args.force_single_worker:
        args.num_workers = 0
        print("Forcing single worker data loading")
    elif args.model_parallel and not args.cache_flows and not args.precache_flows:
        print("\n" + "="*60)
        print("WARNING: Using model parallelism without flow caching!")
        print("This may cause device conflicts with multiple workers.")
        print("Consider adding --cache_flows or --precache_flows")
        print("Setting num_workers=0 for safety")
        print("="*60 + "\n")
        args.num_workers = 0
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create trainer and start training
    trainer = ParallelFusionTrainer(args)
    trainer.train()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()