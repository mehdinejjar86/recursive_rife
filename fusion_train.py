import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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

# Import your dataset and model
from fusion_dataset import create_multi_dataloader
from fusion_model import FusionLoss, create_fusion_model


def setup_distributed():
    """Initialize distributed training if launched with torchrun"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group
        dist.init_process_group(backend='nccl', init_method='env://')
        
        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        print(f"[Rank {rank}/{world_size}] Initialized process on GPU {local_rank}")
        return device, rank, world_size, local_rank, True
    else:
        # Single GPU/CPU mode
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Running in single-device mode on {device}")
        return device, 0, 1, 0, False


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


class MetricsCalculator:
    """Calculate PSNR and SSIM metrics at specified resolution"""
    
    def __init__(self, eval_size=(256, 256)):
        self.eval_size = eval_size
    
    def calculate_psnr(self, pred, target):
        """Calculate PSNR between predicted and target images"""
        # Ensure tensors are on CPU and in numpy format
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Handle batch dimension
        if pred.ndim == 4:  # [B, C, H, W]
            pred = pred.transpose(0, 2, 3, 1)  # [B, H, W, C]
            target = target.transpose(0, 2, 3, 1)
        elif pred.ndim == 3 and pred.shape[0] == 3:  # [C, H, W]
            pred = pred.transpose(1, 2, 0)  # [H, W, C]
            target = target.transpose(1, 2, 0)
        
        # Clamp values to [0, 1]
        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)
        
        # Calculate PSNR
        if pred.ndim == 4:  # Batch
            psnr_values = []
            for i in range(pred.shape[0]):
                psnr_val = psnr_func(target[i], pred[i], data_range=1.0)
                psnr_values.append(psnr_val)
            return np.mean(psnr_values)
        else:  # Single image
            return psnr_func(target, pred, data_range=1.0)
    
    def calculate_ssim(self, pred, target):
        """Calculate SSIM between predicted and target images"""
        # Ensure tensors are on CPU and in numpy format
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Handle batch dimension
        if pred.ndim == 4:  # [B, C, H, W]
            pred = pred.transpose(0, 2, 3, 1)  # [B, H, W, C]
            target = target.transpose(0, 2, 3, 1)
        elif pred.ndim == 3 and pred.shape[0] == 3:  # [C, H, W]
            pred = pred.transpose(1, 2, 0)  # [H, W, C]
            target = target.transpose(1, 2, 0)
        
        # Clamp values to [0, 1]
        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)
        
        # Calculate SSIM
        if pred.ndim == 4:  # Batch
            ssim_values = []
            for i in range(pred.shape[0]):
                # SSIM expects channel_axis
                ssim_val = ssim_func(target[i], pred[i], 
                                    data_range=1.0, 
                                    channel_axis=2 if pred.shape[-1] == 3 else None)
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        else:  # Single image
            return ssim_func(target, pred, 
                           data_range=1.0,
                           channel_axis=2 if pred.shape[-1] == 3 else None)
    
    def resize_for_metrics(self, tensor, size=None):
        """
        Resize tensor to specified size for metric calculation
        
        Args:
            tensor: Input tensor to resize
            size: Target size (H, W). If None, uses self.eval_size
        
        Returns:
            Resized tensor
        """
        # Use instance's eval_size if no size is provided
        if size is None:
            size = self.eval_size
            
        # No resizing needed if already the right size
        if tensor.shape[-2:] == size:
            return tensor
        
        # Use bilinear interpolation for resizing
        resized = torch.nn.functional.interpolate(
            tensor if tensor.dim() == 4 else tensor.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False
        )
        
        return resized if tensor.dim() == 4 else resized.squeeze(0)


class DistributedFusionTrainer:
    """Distributed Trainer for Multi-Anchor Fusion Model"""
    
    def __init__(self, config):
        self.config = config
        
        # Setup distributed training
        self.device, self.rank, self.world_size, self.local_rank, self.is_distributed = setup_distributed()
        
        # Only rank 0 handles directories and logging
        if self.rank == 0:
            # Create timestamp-based run directory
            self.run_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            self.run_name = f"{config.run_name}_{self.run_timestamp}" if hasattr(config, 'run_name') and config.run_name else self.run_timestamp
            
            self.config.checkpoint_dir = Path("runs") / self.run_name / Path(self.config.checkpoint_dir)
            self.config.log_dir = Path("runs") / self.run_name / Path(self.config.log_dir)
            self.config.sample_dir = Path("runs") / self.run_name / Path(self.config.sample_dir)
            
            self.setup_directories()
        else:
            self.run_timestamp = None
            self.run_name = None
        
        # Synchronize before continuing
        if self.is_distributed:
            dist.barrier()
        
        # Create dataloader (distributed-aware)
        self.train_loader, self.train_dataset = self._create_distributed_dataloader()
        
        # Create model
        self.model = self._create_model()
        
        # Wrap model with DDP if distributed
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            if self.rank == 0:
                print(f"Model wrapped with DistributedDataParallel across {self.world_size} GPUs")
        
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
        
        # Setup metrics calculator with configurable size
        metric_size = tuple(config.metric_size) if hasattr(config, 'metric_size') else (256, 256)
        self.metrics_calc = MetricsCalculator(eval_size=metric_size)
        
        # Setup logging (only rank 0)
        if self.rank == 0:
            self.writer = SummaryWriter(self.config.log_dir)
        else:
            self.writer = None
        
        self.start_epoch = 0
        
        # Tracking variables
        self.best_psnr = 0
        self.best_ssim = 0
        self.epoch_metrics = []
        self.path_metrics = {}
        
        # Load checkpoint if exists
        if config.resume:
            self.load_checkpoint(config.resume)
    
    def setup_directories(self):
        """Create directory structure (only rank 0)"""
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.log_dir = Path(self.config.log_dir)
        self.sample_dir = Path(self.config.sample_dir)
        
        # Create all directories
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
        config_dict['world_size'] = self.world_size
        
        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
        
        # Save command
        command_path = self.checkpoint_dir / 'command.txt'
        with open(command_path, 'w') as f:
            import sys
            f.write(' '.join(sys.argv))
    
    def _create_distributed_dataloader(self):
        """Create distributed-aware dataloader"""
        # Parse gt_paths and steps
        gt_paths = self.config.gt_paths
        steps = self.config.steps
        
        # Ensure steps match paths
        if isinstance(steps, int):
            steps = [steps] * len(gt_paths)
        elif len(steps) == 1 and len(gt_paths) > 1:
            steps = steps * len(gt_paths)
        elif len(steps) != len(gt_paths):
            raise ValueError(f"Number of steps ({len(steps)}) must match number of paths ({len(gt_paths)})")
        
        # Get optional parameters
        mix_strategy = getattr(self.config, 'mix_strategy', 'uniform')
        path_weights = getattr(self.config, 'path_weights', None)
        cache_flows = getattr(self.config, 'cache_flows', False)
        
        # Adjust batch size for distributed training
        if self.is_distributed:
            # Each GPU gets batch_size samples
            actual_batch_size = self.config.batch_size
            if self.rank == 0:
                print(f"Distributed training: each GPU processes batch_size={actual_batch_size}")
                print(f"Effective batch size: {actual_batch_size * self.world_size}")
        else:
            actual_batch_size = self.config.batch_size
        
        # Adjust num_workers for MPS
        num_workers = self.config.num_workers
        if self.device.type == 'mps' and num_workers > 0:
            print(f"Warning: MPS device detected. Setting num_workers=0 to avoid multiprocessing issues.")
            num_workers = 0
        
        # Create the dataloader (without distributed-specific args that may not be supported)
        train_dataloader, train_dataset = create_multi_dataloader(
            gt_paths=gt_paths,
            steps=steps,
            anchor=self.config.num_anchors,
            scale=self.config.scale,
            UHD=self.config.UHD,
            batch_size=actual_batch_size,
            shuffle=not self.is_distributed,  # Shuffle handled by DistributedSampler if distributed
            num_workers=num_workers,  # Use adjusted num_workers
            model_dir=self.config.rife_model_dir,
            mix_strategy=mix_strategy,
            path_weights=path_weights,
            cache_flows=cache_flows
        )
        
        # If distributed, wrap with DistributedSampler
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            
            # Recreate dataloader with sampler
            from fusion_dataset import collate_fn
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=actual_batch_size,
                sampler=sampler,
                num_workers=num_workers,  # Use adjusted num_workers
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True  # Important for DDP
            )
        
        # Print dataset statistics (only rank 0)
        if self.rank == 0:
            print("\nTraining Dataset Configuration:")
            print("-" * 50)
            if hasattr(train_dataset, 'get_path_statistics'):
                stats = train_dataset.get_path_statistics()
                for path, info in stats.items():
                    print(f"Path: {path}")
                    print(f"  Step: {info['step']}, Samples: {info['num_samples']} ({info['percentage']:.1f}%)")
            print(f"Total training samples: {len(train_dataset)}")
            print(f"Samples per GPU: {len(train_dataset) // self.world_size}")
            print(f"Mix strategy: {mix_strategy}")
            print("-" * 50)
        
        return train_dataloader, train_dataset
    
    def _create_model(self):
        """Create and initialize model"""
        model = create_fusion_model(
            num_anchors=self.config.num_anchors,
            base_channels=self.config.base_channels
        ).to(self.device)
        
        if self.rank == 0:
            print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
        
        return model
    
    def _create_optimizer(self):
        """Create optimizer"""
        # Use the base model parameters if wrapped with DDP
        model_params = self.model.module.parameters() if hasattr(self.model, 'module') else self.model.parameters()
        
        if self.config.optimizer == 'adam':
            return optim.Adam(
                model_params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(
                model_params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
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
        elif self.config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        # Set epoch for distributed sampler
        if self.is_distributed and hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        # Tracking variables for the epoch
        epoch_losses = {'total': 0, 'l1': 0, 'perceptual': 0, 'consistency': 0}
        epoch_psnr = 0
        epoch_ssim = 0
        num_batches = 0
        
        # Progress bar only on rank 0
        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")
        else:
            pbar = self.train_loader
        
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
            
            # Compute loss
            losses = self.loss_fn(output, I_gt, aux_outputs)
            total_loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                base_model = self.model.module if hasattr(self.model, 'module') else self.model
                nn.utils.clip_grad_norm_(base_model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            # Calculate metrics at specified resolution
            with torch.no_grad():
                # Resize to metric size for consistent calculation
                output_resized = self.metrics_calc.resize_for_metrics(output)
                I_gt_resized = self.metrics_calc.resize_for_metrics(I_gt)
                
                # Calculate PSNR and SSIM
                batch_psnr = self.metrics_calc.calculate_psnr(output_resized, I_gt_resized)
                batch_ssim = self.metrics_calc.calculate_ssim(output_resized, I_gt_resized)
                
                epoch_psnr += batch_psnr
                epoch_ssim += batch_ssim
            
            # Update epoch losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            num_batches += 1
            
            # Update progress bar (rank 0 only)
            if self.rank == 0 and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'psnr': f"{batch_psnr:.2f}",
                    'ssim': f"{batch_ssim:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
        
        # Average metrics for the epoch
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        epoch_psnr /= num_batches
        epoch_ssim /= num_batches
        
        # Synchronize metrics across all GPUs
        if self.is_distributed:
            # Convert to tensors for all_reduce
            loss_tensor = torch.tensor([epoch_losses[key] for key in epoch_losses], device=self.device)
            psnr_tensor = torch.tensor([epoch_psnr], device=self.device)
            ssim_tensor = torch.tensor([epoch_ssim], device=self.device)
            
            # All-reduce
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(psnr_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(ssim_tensor, op=dist.ReduceOp.SUM)
            
            # Average across GPUs
            loss_tensor /= self.world_size
            psnr_tensor /= self.world_size
            ssim_tensor /= self.world_size
            
            # Convert back to dict
            for i, key in enumerate(epoch_losses):
                epoch_losses[key] = loss_tensor[i].item()
            epoch_psnr = psnr_tensor.item()
            epoch_ssim = ssim_tensor.item()
        
        # Add metrics to losses dict
        epoch_losses['psnr'] = epoch_psnr
        epoch_losses['ssim'] = epoch_ssim
        
        return epoch_losses
    
    def validate(self):
        """Validate the model (simplified for distributed)"""
        # For simplicity, we'll skip validation in distributed mode
        # You can implement distributed validation similar to training
        return None
    
    def log_epoch(self, epoch, train_losses, val_losses=None):
        """Log epoch metrics to tensorboard (rank 0 only)"""
        if self.rank != 0:
            return
        
        # Log training metrics
        for name, value in train_losses.items():
            self.writer.add_scalar(f'train/{name}', value, epoch)
        
        # Log validation metrics if available
        if val_losses:
            for name, value in val_losses.items():
                self.writer.add_scalar(f'val/{name}', value, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', current_lr, epoch)
        
        # Store metrics for later analysis
        metrics_entry = {
            'epoch': epoch,
            'train': train_losses,
            'val': val_losses if val_losses else None,
            'lr': current_lr
        }
        self.epoch_metrics.append(metrics_entry)
        
        # Save metrics to JSON
        metrics_path = self.checkpoint_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2, default=str)
    
    def save_samples(self, output, target, I0, I1, epoch):
        """Save sample predictions (rank 0 only)"""
        if self.rank != 0:
            return
        
        # Move to CPU and denormalize if needed
        output = output.detach().cpu().clamp(0, 1)
        target = target.detach().cpu().clamp(0, 1)
        I0 = I0.detach().cpu().clamp(0, 1)
        I1 = I1.detach().cpu().clamp(0, 1)
        
        # Resize all to metric size for consistent visualization
        metric_size = self.metrics_calc.eval_size
        
        output = self.metrics_calc.resize_for_metrics(output, metric_size)
        target = self.metrics_calc.resize_for_metrics(target, metric_size)
        I0 = self.metrics_calc.resize_for_metrics(I0, metric_size)
        I1 = self.metrics_calc.resize_for_metrics(I1, metric_size)
        
        # Save first sample in batch
        sample_idx = 0
        
        # Create grid - stack images horizontally
        from torchvision.utils import save_image, make_grid
        
        # Get individual images
        i0_img = I0[sample_idx:sample_idx+1]
        output_img = output[sample_idx:sample_idx+1]
        target_img = target[sample_idx:sample_idx+1]
        i1_img = I1[sample_idx:sample_idx+1]
        
        # Stack them for grid
        grid_images = torch.cat([i0_img, output_img, target_img, i1_img], dim=0)
        
        # Save grid to file
        save_path = self.sample_dir / f'epoch_{epoch:04d}.png'
        save_image(grid_images, save_path, nrow=4, normalize=False)
        
        # For tensorboard, use make_grid to create a proper grid
        grid_for_tb = make_grid(grid_images, nrow=4, normalize=False)
        self.writer.add_image('samples/prediction', grid_for_tb, epoch)
        
        # Calculate and display metrics for this sample
        sample_psnr = self.metrics_calc.calculate_psnr(
            output[sample_idx], target[sample_idx]
        )
        sample_ssim = self.metrics_calc.calculate_ssim(
            output[sample_idx], target[sample_idx]
        )
        
        print(f"  Sample PSNR: {sample_psnr:.2f} dB, SSIM: {sample_ssim:.4f}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint (rank 0 only)"""
        if self.rank != 0:
            return
        
        # Get base model state dict
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'epoch_metrics': self.epoch_metrics,
            'config': vars(self.config),
            'run_name': self.run_name,
            'run_timestamp': self.run_timestamp,
            'world_size': self.world_size
        }
        
        # Save regular checkpoint every N epochs
        if epoch % self.config.save_interval == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  Best model saved!")
        
        # Always save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        # Map to current device
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank} if self.is_distributed else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state
        base_model = self.model.module if hasattr(self.model, 'module') else self.model
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_psnr = checkpoint.get('best_psnr', 0)
        self.best_ssim = checkpoint.get('best_ssim', 0)
        self.epoch_metrics = checkpoint.get('epoch_metrics', [])
        
        if self.rank == 0:
            print(f"Checkpoint loaded from {checkpoint_path}")
            print(f"Resuming from epoch {self.start_epoch}")
            print(f"Best PSNR: {self.best_psnr:.2f}, Best SSIM: {self.best_ssim:.4f}")
    
    def train(self):
        """Main training loop"""
        if self.rank == 0:
            print(f"\nStarting distributed training for {self.config.epochs} epochs")
            print(f"World size: {self.world_size} GPUs")
            print(f"Training samples per GPU: {len(self.train_dataset) // self.world_size}")
            print(f"Batch size per GPU: {self.config.batch_size}")
            print(f"Effective batch size: {self.config.batch_size * self.world_size}")
            print(f"Total batches per epoch: {len(self.train_loader)}")
            print(f"Metrics computed at: {self.metrics_calc.eval_size} resolution")
            print("-" * 60)
        
        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                epoch_start_time = datetime.now()
                
                # Train
                train_losses = self.train_epoch(epoch)
                
                # Validate (simplified for now)
                val_losses = self.validate()
                
                # Check if best model (based on training PSNR for simplicity)
                is_best = train_losses['psnr'] > self.best_psnr
                if is_best:
                    self.best_psnr = train_losses['psnr']
                    self.best_ssim = train_losses['ssim']
                    if self.rank == 0:
                        print(f"  New best model! PSNR: {self.best_psnr:.2f} dB, SSIM: {self.best_ssim:.4f}")
                
                # Update scheduler
                if self.scheduler:
                    if self.config.scheduler == 'plateau' and val_losses:
                        self.scheduler.step(val_losses['total'])
                    else:
                        self.scheduler.step()
                
                # Log epoch metrics (rank 0 only)
                self.log_epoch(epoch, train_losses, val_losses)
                
                # Print epoch summary (rank 0 only)
                if self.rank == 0:
                    epoch_time = (datetime.now() - epoch_start_time).total_seconds()
                    print(f"\n{'='*60}")
                    print(f"Epoch {epoch}/{self.config.epochs} Summary (Time: {epoch_time:.1f}s)")
                    print(f"  Train - Loss: {train_losses['total']:.4f}, PSNR: {train_losses['psnr']:.2f} dB, SSIM: {train_losses['ssim']:.4f}")
                    if val_losses:
                        print(f"  Val   - Loss: {val_losses['total']:.4f}, PSNR: {val_losses['psnr']:.2f} dB, SSIM: {val_losses['ssim']:.4f}")
                    print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                    print(f"  Best PSNR: {self.best_psnr:.2f} dB, Best SSIM: {self.best_ssim:.4f}")
                
                # Save sample predictions (rank 0 only)
                if self.rank == 0 and epoch % self.config.sample_interval == 0:
                    # Get a sample batch for visualization
                    sample_batch = next(iter(self.train_loader))
                    with torch.no_grad():
                        I0_all = sample_batch['I0'].to(self.device)
                        I1_all = sample_batch['I1'].to(self.device)
                        flows_all = sample_batch['flows'].to(self.device)
                        masks_all = sample_batch['masks'].to(self.device)
                        timesteps = sample_batch['timesteps'].to(self.device)
                        I_gt = sample_batch['I_gt'].to(self.device)
                        
                        output, _ = self.model(I0_all, I1_all, flows_all, masks_all, timesteps)
                        self.save_samples(output, I_gt, I0_all[:, 0], I1_all[:, -1], epoch)
                
                # Save checkpoint (rank 0 only)
                self.save_checkpoint(epoch, is_best)
                
                if self.rank == 0:
                    print("="*60)
                
                # Synchronize all processes before next epoch
                if self.is_distributed:
                    dist.barrier()
        
        finally:
            # Save final model
            if self.rank == 0:
                self.save_checkpoint(self.config.epochs, False)
                if self.writer:
                    self.writer.close()
                
                print("\n" + "="*60)
                print("Training completed!")
                print(f"Best PSNR: {self.best_psnr:.2f} dB")
                print(f"Best SSIM: {self.best_ssim:.4f}")
                print(f"Run directory: {self.run_name}")
                print("="*60)
            
            # Clean up distributed training
            cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Anchor Fusion Model with Distributed Support')
    
    # Dataset arguments
    parser.add_argument('--gt_paths', '--gt_path', type=str, nargs='+', required=True,
                        dest='gt_paths',
                        help='Path(s) to ground truth images (can specify one or multiple)')
    parser.add_argument('--steps', '--step', type=int, nargs='+', required=True,
                        dest='steps',
                        help='Step size(s) for sequence generation (single value or one per path)')
    
    # Validation data arguments
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (0.0 to disable, default: 0.1)')
    parser.add_argument('--val_paths', type=str, nargs='+', default=None,
                        help='Separate validation dataset paths (optional)')
    parser.add_argument('--val_steps', type=int, nargs='+', default=None,
                        help='Step sizes for validation paths')
    parser.add_argument('--mix_strategy', type=str, default='uniform',
                        choices=['uniform', 'weighted', 'sequential', 'balanced'],
                        help='How to mix samples from different paths')
    parser.add_argument('--path_weights', type=float, nargs='+',
                        help='Weights for each path (for weighted strategy)')
    parser.add_argument('--cache_flows', action='store_true',
                        help='Cache extracted flows to speed up training')
    
    # Model arguments
    parser.add_argument('--num_anchors', type=int, default=3,
                        help='Number of anchor frames')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels in model')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for processing')
    parser.add_argument('--UHD', action='store_true',
                        help='Support for 4K images')
    parser.add_argument('--rife_model_dir', type=str, default='ckpt/rifev4_25',
                        help='Path to RIFE model directory')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size per GPU for training')
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
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    
    # Logging and saving
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Base directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Base directory for tensorboard logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Base directory for sample predictions')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional name for this training run')
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='Sample saving interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint saving interval (epochs)')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validation interval (epochs)')
    
    # Metrics
    parser.add_argument('--metric_size', type=int, nargs=2, default=[256, 256],
                        help='Size for computing metrics (H W)')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers per GPU')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Distributed training arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (set automatically by torchrun)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create trainer and start training
    trainer = DistributedFusionTrainer(args)
    trainer.train()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()