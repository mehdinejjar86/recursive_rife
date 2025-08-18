#!/usr/bin/env python3
"""Test script to verify model dimensions and functionality with parallelism support"""

import torch
import torch.nn as nn
from fusion_model import create_fusion_model, MultiAnchorFusionModel
import argparse


class ModelParallelWrapper(nn.Module):
    """
    Wrapper to enable model parallelism for testing large images
    """
    
    def __init__(self, base_model, num_gpus=None, split_strategy='spatial'):
        super().__init__()
        
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.split_strategy = split_strategy
        
        if self.num_gpus <= 1:
            # Single GPU - no parallelism needed
            self.model = base_model
            self.parallel_mode = False
            print(f"Using single device (no parallelism)")
        else:
            # Multiple GPUs - enable parallelism
            self.models = nn.ModuleList()
            for i in range(self.num_gpus):
                # Create a model copy on each GPU
                model_copy = create_fusion_model(
                    num_anchors=base_model.num_anchors,
                    base_channels=base_model.base_channels
                )
                model_copy = model_copy.cuda(i)
                self.models.append(model_copy)
            self.parallel_mode = True
            
            # Sync parameters across GPUs
            if self.num_gpus > 1:
                for i in range(1, self.num_gpus):
                    for param_main, param_copy in zip(
                        self.models[0].parameters(), 
                        self.models[i].parameters()
                    ):
                        param_copy.data = param_main.data
            
            print(f"Model Parallel: Using {self.num_gpus} GPUs with {split_strategy} strategy")
    
    def forward(self, I0_all, I1_all, flows_all, masks_all, timesteps):
        """Forward pass with optional model parallelism"""
        
        if not self.parallel_mode:
            return self.model(I0_all, I1_all, flows_all, masks_all, timesteps)
        
        B, A, C, H, W = I0_all.shape
        
        if self.split_strategy == 'spatial':
            return self._forward_spatial_parallel(
                I0_all, I1_all, flows_all, masks_all, timesteps
            )
        else:
            # Default to batch splitting
            return self._forward_batch_parallel(
                I0_all, I1_all, flows_all, masks_all, timesteps
            )
    
    def _forward_spatial_parallel(self, I0_all, I1_all, flows_all, masks_all, timesteps):
        """Split images spatially across GPUs"""
        B, A, C, H, W = I0_all.shape
        
        # Calculate split size with overlap
        overlap = 32
        base_strip_height = H // self.num_gpus
        
        if base_strip_height < 64:
            # Image too small for spatial splitting
            print(f"  Image height {H} too small for {self.num_gpus} GPUs, using single GPU")
            return self.models[0](
                I0_all.cuda(0), I1_all.cuda(0), 
                flows_all.cuda(0), masks_all.cuda(0), 
                timesteps.cuda(0)
            )
        
        outputs = []
        aux_outputs_list = []
        
        print(f"  Splitting {H}x{W} image into {self.num_gpus} strips")
        
        for gpu_id in range(self.num_gpus):
            # Calculate strip boundaries with overlap
            start_h = max(0, gpu_id * base_strip_height - overlap // 2)
            end_h = min(H, (gpu_id + 1) * base_strip_height + overlap // 2)
            
            # Extract strip and move to appropriate GPU
            I0_strip = I0_all[:, :, :, start_h:end_h, :].cuda(gpu_id)
            I1_strip = I1_all[:, :, :, start_h:end_h, :].cuda(gpu_id)
            flows_strip = flows_all[:, :, :, start_h:end_h, :].cuda(gpu_id)
            masks_strip = masks_all[:, :, start_h:end_h, :].cuda(gpu_id)
            timesteps_gpu = timesteps.cuda(gpu_id)
            
            # Process strip on its GPU
            with torch.cuda.device(gpu_id):
                output_strip, aux_strip = self.models[gpu_id](
                    I0_strip, I1_strip, flows_strip, masks_strip, timesteps_gpu
                )
            
            outputs.append(output_strip)
            aux_outputs_list.append(aux_strip)
            
            # Report memory per GPU
            mem_mb = torch.cuda.memory_allocated(gpu_id) / 1024**2
            print(f"    GPU {gpu_id}: Strip [{start_h}:{end_h}] - Memory: {mem_mb:.1f} MB")
        
        # Merge outputs
        merged_output = self._merge_spatial_outputs(outputs, H, overlap, base_strip_height)
        aux_outputs = aux_outputs_list[0]  # Use first GPU's aux outputs
        
        return merged_output, aux_outputs
    
    def _forward_batch_parallel(self, I0_all, I1_all, flows_all, masks_all, timesteps):
        """Split batch across GPUs"""
        B = I0_all.shape[0]
        
        if B < self.num_gpus:
            # Batch smaller than GPU count, use single GPU
            return self.models[0](
                I0_all.cuda(0), I1_all.cuda(0), 
                flows_all.cuda(0), masks_all.cuda(0), 
                timesteps.cuda(0)
            )
        
        samples_per_gpu = B // self.num_gpus + (1 if B % self.num_gpus else 0)
        
        outputs = []
        aux_outputs_list = []
        
        for gpu_id in range(self.num_gpus):
            start_b = gpu_id * samples_per_gpu
            end_b = min(start_b + samples_per_gpu, B)
            
            if start_b >= B:
                break
            
            # Extract batch for this GPU
            I0_gpu = I0_all[start_b:end_b].cuda(gpu_id)
            I1_gpu = I1_all[start_b:end_b].cuda(gpu_id)
            flows_gpu = flows_all[start_b:end_b].cuda(gpu_id)
            masks_gpu = masks_all[start_b:end_b].cuda(gpu_id)
            timesteps_gpu = timesteps[start_b:end_b].cuda(gpu_id)
            
            # Process on GPU
            with torch.cuda.device(gpu_id):
                output_gpu, aux_gpu = self.models[gpu_id](
                    I0_gpu, I1_gpu, flows_gpu, masks_gpu, timesteps_gpu
                )
            
            outputs.append(output_gpu.cuda(0))
            aux_outputs_list.append(aux_gpu)
        
        # Concatenate outputs
        merged_output = torch.cat(outputs, dim=0)
        
        # Merge auxiliary outputs
        aux_outputs = {}
        for key in aux_outputs_list[0].keys():
            if isinstance(aux_outputs_list[0][key], list):
                aux_outputs[key] = aux_outputs_list[0][key]
            else:
                aux_values = [aux[key].cuda(0) for aux in aux_outputs_list]
                aux_outputs[key] = torch.cat(aux_values, dim=0)
        
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
            
            # Create weight for blending
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


def test_model_forward(use_parallel=False, parallel_strategy='spatial'):
    """Test the model forward pass with correct dimensions"""
    print("=" * 50)
    print("Testing Multi-Anchor Fusion Model")
    if use_parallel and torch.cuda.device_count() > 1:
        print(f"Mode: Model Parallel ({parallel_strategy})")
    else:
        print("Mode: Single Device")
    print("=" * 50)
    
    # Setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: CUDA ({torch.cuda.device_count()} GPU(s) available)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"Device: CPU")
    print()
    
    # Model parameters
    batch_size = 2
    num_anchors = 3
    base_channels = 64
    height, width = 256, 256
    
    print(f"Test Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of anchors: {num_anchors}")
    print(f"  Base channels: {base_channels}")
    print(f"  Input size: {height}x{width}")
    print()
    
    # Create model
    print("Creating model...")
    base_model = create_fusion_model(num_anchors=num_anchors, base_channels=base_channels)
    
    if use_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = ModelParallelWrapper(base_model, split_strategy=parallel_strategy)
    else:
        model = base_model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Create dummy inputs
    print("Creating test inputs...")
    I0_all = torch.randn(batch_size, num_anchors, 3, height, width)
    I1_all = torch.randn(batch_size, num_anchors, 3, height, width)
    flows_all = torch.randn(batch_size, num_anchors, 4, height, width)
    masks_all = torch.rand(batch_size, num_anchors, height, width)
    timesteps = torch.rand(batch_size, num_anchors)
    
    # Move to device if not using parallel
    if not (use_parallel and torch.cuda.device_count() > 1):
        I0_all = I0_all.to(device)
        I1_all = I1_all.to(device)
        flows_all = flows_all.to(device)
        masks_all = masks_all.to(device)
        timesteps = timesteps.to(device)
    
    print("Input shapes:")
    print(f"  I0_all: {I0_all.shape}")
    print(f"  I1_all: {I1_all.shape}")
    print(f"  flows_all: {flows_all.shape}")
    print(f"  masks_all: {masks_all.shape}")
    print(f"  timesteps: {timesteps.shape}")
    print()
    
    # Test forward pass
    print("Running forward pass...")
    try:
        with torch.no_grad():
            output, aux_outputs = model(I0_all, I1_all, flows_all, masks_all, timesteps)
        
        print("✓ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print()
        
        print("Auxiliary outputs:")
        for key, value in aux_outputs.items():
            if isinstance(value, list):
                print(f"  {key}: list of {len(value)} tensors, first shape: {value[0].shape}")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        print()
        
        # Memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if torch.cuda.memory_allocated(i) > 0:
                    print(f"GPU {i} memory: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB allocated, "
                          f"{torch.cuda.memory_reserved(i) / 1024**2:.2f} MB reserved")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_sizes(use_parallel=False, parallel_strategy='spatial'):
    """Test model with different input sizes"""
    print("\n" + "=" * 50)
    print("Testing Different Input Sizes")
    if use_parallel and torch.cuda.device_count() > 1:
        print(f"Mode: Model Parallel ({parallel_strategy})")
    else:
        print("Mode: Single Device")
    print("=" * 50)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using: {torch.cuda.device_count()} GPU(s)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using: MPS")
    else:
        device = torch.device('cpu')
        print(f"Using: CPU")
    print()
    
    # Create model
    base_model = create_fusion_model(num_anchors=3, base_channels=64)
    
    if use_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = ModelParallelWrapper(base_model, split_strategy=parallel_strategy)
    else:
        model = base_model.to(device)
    
    model.eval()
    
    test_sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    batch_size = 1
    num_anchors = 3
    
    for h, w in test_sizes:
        print(f"\nTesting {h}x{w}...")
        
        # Clear cache before each test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        I0_all = torch.randn(batch_size, num_anchors, 3, h, w)
        I1_all = torch.randn(batch_size, num_anchors, 3, h, w)
        flows_all = torch.randn(batch_size, num_anchors, 4, h, w)
        masks_all = torch.rand(batch_size, num_anchors, h, w)
        timesteps = torch.rand(batch_size, num_anchors)
        
        # Move to device if not using parallel
        if not (use_parallel and torch.cuda.device_count() > 1):
            I0_all = I0_all.to(device)
            I1_all = I1_all.to(device)
            flows_all = flows_all.to(device)
            masks_all = masks_all.to(device)
            timesteps = timesteps.to(device)
        
        try:
            with torch.no_grad():
                output, _ = model(I0_all, I1_all, flows_all, masks_all, timesteps)
            
            print(f"  ✓ Size {h}x{w} passed. Output shape: {output.shape}")
            
            # Report total memory usage
            if torch.cuda.is_available():
                total_mem = 0
                for i in range(torch.cuda.device_count()):
                    if torch.cuda.memory_allocated(i) > 0:
                        mem_gb = torch.cuda.memory_allocated(i) / 1024**3
                        total_mem += mem_gb
                        if use_parallel and torch.cuda.device_count() > 1:
                            print(f"    GPU {i}: {mem_gb:.2f} GB")
                if torch.cuda.device_count() > 1:
                    print(f"    Total: {total_mem:.2f} GB across all GPUs")
                else:
                    print(f"    Memory: {total_mem:.2f} GB")
                    
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ❌ Size {h}x{w} failed: Out of Memory")
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        if torch.cuda.memory_allocated(i) > 0:
                            mem_gb = torch.cuda.max_memory_allocated(i) / 1024**3
                            print(f"    GPU {i} peak memory: {mem_gb:.2f} GB")
                break
            else:
                print(f"  ❌ Size {h}x{w} failed: {e}")
        except Exception as e:
            print(f"  ❌ Size {h}x{w} failed: {e}")
        
        # Clear memory after each test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_gradient_flow(use_parallel=False):
    """Test that gradients flow properly through the model"""
    print("\n" + "=" * 50)
    print("Testing Gradient Flow")
    print("=" * 50)
    
    # Use CPU for gradient testing to avoid memory issues
    device = torch.device('cpu')
    model = create_fusion_model(num_anchors=3, base_channels=32).to(device)
    model.train()
    
    # Small inputs for gradient testing
    batch_size = 1
    num_anchors = 3
    h, w = 64, 64
    
    I0_all = torch.randn(batch_size, num_anchors, 3, h, w, requires_grad=True).to(device)
    I1_all = torch.randn(batch_size, num_anchors, 3, h, w, requires_grad=True).to(device)
    flows_all = torch.randn(batch_size, num_anchors, 4, h, w, requires_grad=True).to(device)
    masks_all = torch.rand(batch_size, num_anchors, h, w, requires_grad=True).to(device)
    timesteps = torch.rand(batch_size, num_anchors, requires_grad=True).to(device)
    
    # Forward pass
    output, aux_outputs = model(I0_all, I1_all, flows_all, masks_all, timesteps)
    
    # Create dummy loss
    target = torch.randn_like(output)
    loss = nn.functional.mse_loss(output, target)
    
    print(f"Loss value: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print("\nGradient checks:")
    
    # Check model parameters
    params_with_grad = 0
    params_without_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad += 1
        else:
            params_without_grad += 1
    
    print(f"  Parameters with gradients: {params_with_grad}")
    print(f"  Parameters without gradients: {params_without_grad}")
    
    # Check input gradients
    if I0_all.grad is not None:
        print(f"  ✓ I0_all has gradients. Mean abs grad: {I0_all.grad.abs().mean().item():.6f}")
    if flows_all.grad is not None:
        print(f"  ✓ flows_all has gradients. Mean abs grad: {flows_all.grad.abs().mean().item():.6f}")
    if timesteps.grad is not None:
        print(f"  ✓ timesteps has gradients. Mean abs grad: {timesteps.grad.abs().mean().item():.6f}")
    
    if params_with_grad > 0:
        print("\n✅ Gradient flow test passed!")
    else:
        print("\n❌ No gradients found in model parameters!")


def main():
    parser = argparse.ArgumentParser(description='Test Fusion Model with optional parallelism')
    parser.add_argument('--model_parallel', action='store_true',
                        help='Enable model parallelism for large images')
    parser.add_argument('--parallel_strategy', type=str, default='spatial',
                        choices=['spatial', 'batch'],
                        help='Parallelism strategy: spatial (split image) or batch')
    parser.add_argument('--skip_gradient_test', action='store_true',
                        help='Skip gradient flow test')
    
    args = parser.parse_args()
    
    # Print system info
    print("System Information:")
    print("-" * 50)
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"MPS available: Yes (Apple Silicon)")
    else:
        print(f"Using: CPU")
    print()
    
    # Determine if we should use parallelism
    use_parallel = args.model_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1
    
    if args.model_parallel and not use_parallel:
        print("WARNING: Model parallelism requested but not available")
        print("  (Requires multiple CUDA GPUs)")
        print()
    
    # Run tests
    success = test_model_forward(use_parallel, args.parallel_strategy)
    
    if success:
        test_different_sizes(use_parallel, args.parallel_strategy)
        
        if not args.skip_gradient_test:
            test_gradient_flow(use_parallel)
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("=" * 50)
    
    if use_parallel:
        print("\nModel Parallelism Summary:")
        print(f"  Strategy: {args.parallel_strategy}")
        print(f"  GPUs used: {torch.cuda.device_count()}")
        print("  Benefit: Can process larger images by pooling GPU memory")


if __name__ == "__main__":
    main()