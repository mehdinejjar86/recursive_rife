#!/usr/bin/env python3
"""Test script to verify model dimensions and functionality with multi-GPU support"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from fusion_model import create_fusion_model, MultiAnchorFusionModel

def setup_distributed():
    """Initialize distributed training environment"""
    # Check if launched with torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize the process group
        dist.init_process_group(backend='nccl')
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        print(f"[Rank {rank}/{world_size}] Initialized on GPU {local_rank}")
        return device, rank, world_size, local_rank
    else:
        # Fallback to single GPU/CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 
                              'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                              'cpu')
        print(f"Running in single-device mode on {device}")
        return device, 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def test_model_forward_distributed():
    """Test the model forward pass with distributed support"""
    device, rank, world_size, local_rank = setup_distributed()
    
    # Only print from rank 0 to avoid duplicate outputs
    if rank == 0:
        print("=" * 50)
        print("Testing Multi-Anchor Fusion Model (Distributed)")
        print("=" * 50)
        print(f"Total GPUs: {world_size}")
        print()
    
    # Synchronize all processes
    if dist.is_initialized():
        dist.barrier()
    
    # Model parameters
    batch_size = 2  # Per GPU batch size
    num_anchors = 3
    base_channels = 64
    height, width = 256, 256
    
    if rank == 0:
        print(f"Test Configuration:")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Total batch size: {batch_size * world_size}")
        print(f"  Number of anchors: {num_anchors}")
        print(f"  Base channels: {base_channels}")
        print(f"  Input size: {height}x{width}")
        print()
    
    # Create model
    if rank == 0:
        print("Creating model...")
    
    model = create_fusion_model(num_anchors=num_anchors, base_channels=base_channels).to(device)
    
    # Wrap model with DDP if distributed
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if rank == 0:
            print(f"Model wrapped with DistributedDataParallel")
    
    # Count parameters (only on rank 0)
    if rank == 0:
        base_model = model.module if hasattr(model, 'module') else model
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print()
    
    # Create dummy inputs (each GPU gets different data)
    if rank == 0:
        print("Creating test inputs...")
    
    # Add seed offset based on rank for different data per GPU
    torch.manual_seed(42 + rank)
    
    I0_all = torch.randn(batch_size, num_anchors, 3, height, width).to(device)
    I1_all = torch.randn(batch_size, num_anchors, 3, height, width).to(device)
    flows_all = torch.randn(batch_size, num_anchors, 4, height, width).to(device)
    masks_all = torch.rand(batch_size, num_anchors, height, width).to(device)
    timesteps = torch.rand(batch_size, num_anchors).to(device)
    
    if rank == 0:
        print("Input shapes (per GPU):")
        print(f"  I0_all: {I0_all.shape}")
        print(f"  I1_all: {I1_all.shape}")
        print(f"  flows_all: {flows_all.shape}")
        print(f"  masks_all: {masks_all.shape}")
        print(f"  timesteps: {timesteps.shape}")
        print()
    
    # Test forward pass
    if rank == 0:
        print("Running forward pass...")
    
    try:
        with torch.no_grad():
            output, aux_outputs = model(I0_all, I1_all, flows_all, masks_all, timesteps)
        
        # Gather statistics from all GPUs
        output_min = output.min().item()
        output_max = output.max().item()
        
        if dist.is_initialized():
            # Gather min/max from all ranks
            min_tensor = torch.tensor([output_min], device=device)
            max_tensor = torch.tensor([output_max], device=device)
            dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
            output_min = min_tensor.item()
            output_max = max_tensor.item()
        
        if rank == 0:
            print("✓ Forward pass successful!")
            print(f"Output shape (per GPU): {output.shape}")
            print(f"Output range (global): [{output_min:.3f}, {output_max:.3f}]")
            print()
            
            print("Auxiliary outputs:")
            for key, value in aux_outputs.items():
                if isinstance(value, list):
                    print(f"  {key}: list of {len(value)} tensors, first shape: {value[0].shape}")
                elif isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
            print()
        
        # Memory usage per GPU
        if torch.cuda.is_available() and device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
            
            print(f"[GPU {local_rank}] Memory allocated: {mem_allocated:.2f} MB, Reserved: {mem_reserved:.2f} MB")
            
            if dist.is_initialized():
                dist.barrier()
        
        if rank == 0:
            print("\n✅ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ Forward pass failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        cleanup_distributed()

def test_gradient_flow_distributed():
    """Test gradient flow in distributed setting"""
    device, rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("\n" + "=" * 50)
        print("Testing Gradient Flow (Distributed)")
        print("=" * 50)
    
    model = create_fusion_model(num_anchors=3, base_channels=32).to(device)
    
    # Wrap with DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
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
    
    # Backward pass
    loss.backward()
    
    # Gather loss from all ranks
    loss_value = loss.item()
    if dist.is_initialized():
        loss_tensor = torch.tensor([loss_value], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_value = loss_tensor.item() / world_size
    
    if rank == 0:
        print(f"Average loss across GPUs: {loss_value:.4f}")
        
        # Check gradients
        print("\nGradient checks:")
        
        base_model = model.module if hasattr(model, 'module') else model
        params_with_grad = sum(1 for p in base_model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        total_params = sum(1 for _ in base_model.parameters())
        
        print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
        
        if I0_all.grad is not None:
            print(f"  ✓ I0_all has gradients. Mean abs grad: {I0_all.grad.abs().mean().item():.6f}")
        if flows_all.grad is not None:
            print(f"  ✓ flows_all has gradients. Mean abs grad: {flows_all.grad.abs().mean().item():.6f}")
        
        if params_with_grad > 0:
            print("\n✅ Gradient flow test passed!")
        else:
            print("\n❌ No gradients found in model parameters!")
    
    cleanup_distributed()

def test_different_sizes():
    """Test model with different input sizes"""
    print("\n" + "=" * 50)
    print("Testing Different Input Sizes")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                          'cpu')
    print(f"Using device: {device}\n")
    
    model = create_fusion_model(num_anchors=3, base_channels=32).to(device)
    model.eval()
    
    test_sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    batch_size = 1
    num_anchors = 3
    
    for h, w in test_sizes:
        print(f"\nTesting {h}x{w}...")
        
        I0_all = torch.randn(batch_size, num_anchors, 3, h, w).to(device)
        I1_all = torch.randn(batch_size, num_anchors, 3, h, w).to(device)
        flows_all = torch.randn(batch_size, num_anchors, 4, h, w).to(device)
        masks_all = torch.rand(batch_size, num_anchors, h, w).to(device)
        timesteps = torch.rand(batch_size, num_anchors).to(device)
        
        try:
            with torch.no_grad():
                output, _ = model(I0_all, I1_all, flows_all, masks_all, timesteps)
            print(f"  ✓ Size {h}x{w} passed. Output shape: {output.shape}")
            
            # Report memory usage for larger sizes
            if torch.cuda.is_available() and device.type == 'cuda' and h >= 512:
                mem_gb = torch.cuda.memory_allocated(device) / 1024**3
                print(f"    Memory used: {mem_gb:.2f} GB")
                
        except Exception as e:
            print(f"  ❌ Size {h}x{w} failed: {e}")
            # If we run out of memory, stop testing larger sizes
            if "out of memory" in str(e).lower():
                print("    Stopping size tests due to memory limitations")
                break

def test_different_sizes_distributed():
    """Test model with different input sizes in distributed setting"""
    device, rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("\n" + "=" * 50)
        print("Testing Different Input Sizes (Distributed)")
        print("=" * 50)
        print(f"Running on {world_size} GPU(s)\n")
    
    model = create_fusion_model(num_anchors=3, base_channels=32).to(device)
    
    # Wrap with DDP if distributed
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    model.eval()
    
    test_sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    batch_size = 1  # Per GPU
    num_anchors = 3
    
    for h, w in test_sizes:
        if rank == 0:
            print(f"\nTesting {h}x{w}...")
        
        # Each GPU gets its own input
        torch.manual_seed(42 + rank)
        I0_all = torch.randn(batch_size, num_anchors, 3, h, w).to(device)
        I1_all = torch.randn(batch_size, num_anchors, 3, h, w).to(device)
        flows_all = torch.randn(batch_size, num_anchors, 4, h, w).to(device)
        masks_all = torch.rand(batch_size, num_anchors, h, w).to(device)
        timesteps = torch.rand(batch_size, num_anchors).to(device)
        
        try:
            with torch.no_grad():
                output, _ = model(I0_all, I1_all, flows_all, masks_all, timesteps)
            
            # Gather success status from all ranks
            success = torch.tensor([1.0], device=device)
            if dist.is_initialized():
                dist.all_reduce(success, op=dist.ReduceOp.MIN)
            
            if success.item() > 0:
                if rank == 0:
                    print(f"  ✓ Size {h}x{w} passed. Output shape per GPU: {output.shape}")
                
                # Report memory usage
                if torch.cuda.is_available() and device.type == 'cuda' and h >= 512:
                    mem_gb = torch.cuda.memory_allocated(device) / 1024**3
                    max_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                    
                    # Gather max memory from all GPUs
                    if dist.is_initialized():
                        mem_tensor = torch.tensor([mem_gb], device=device)
                        max_mem_tensor = torch.tensor([max_mem_gb], device=device)
                        dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX)
                        dist.all_reduce(max_mem_tensor, op=dist.ReduceOp.MAX)
                        mem_gb = mem_tensor.item()
                        max_mem_gb = max_mem_tensor.item()
                    
                    if rank == 0:
                        print(f"    Current memory: {mem_gb:.2f} GB, Peak memory: {max_mem_gb:.2f} GB")
            
        except Exception as e:
            # Signal failure to other ranks
            failure = torch.tensor([0.0], device=device)
            if dist.is_initialized():
                dist.all_reduce(failure, op=dist.ReduceOp.MIN)
            
            if rank == 0:
                print(f"  ❌ Size {h}x{w} failed: {e}")
                if "out of memory" in str(e).lower():
                    print("    Stopping size tests due to memory limitations")
            break
        
        # Synchronize before next size
        if dist.is_initialized():
            dist.barrier()
    
    cleanup_distributed()

def main():
    """Main entry point"""
    # Check if we're in distributed mode
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    if is_distributed:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        world_size = 1
    
    # Run main forward pass test
    success = test_model_forward_distributed()
    
    if success:
        # Run size tests
        if is_distributed and world_size > 1:
            # Test sizes with distributed setup
            test_different_sizes_distributed()
        elif rank == 0:
            # Single GPU/CPU size test
            test_different_sizes()
        
        # Run gradient flow test
        if is_distributed:
            test_gradient_flow_distributed()
        elif rank == 0:
            # Need to setup the model again for gradient test in single mode
            print("\n" + "=" * 50)
            print("Testing Gradient Flow (Single Device)")
            print("=" * 50)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 
                                  'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                                  'cpu')
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
            
            params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
            total_params = sum(1 for _ in model.parameters())
            
            print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
            
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
    
    if rank == 0:
        print("\n" + "=" * 50)
        print("Testing Complete!")
        print("=" * 50)

if __name__ == "__main__":
    main()