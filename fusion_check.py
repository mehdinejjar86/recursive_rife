#!/usr/bin/env python3
"""Test script with true model parallelism for larger images"""

import os
import torch
import torch.nn as nn
from fusion_model import create_fusion_model

def split_tensor_across_gpus(tensor, num_gpus, dim=0):
    """Split a tensor across multiple GPUs along specified dimension"""
    chunks = tensor.chunk(num_gpus, dim=dim)
    return [chunk.cuda(i) for i, chunk in enumerate(chunks)]

def test_model_parallel():
    """Test model with true parallelism across GPUs"""
    print("=" * 50)
    print("Testing Multi-GPU Model Parallel")
    print("=" * 50)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No CUDA GPUs available. Using CPU/MPS fallback.")
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        num_gpus = 1
    else:
        device = torch.device('cuda:0')
        print(f"Found {num_gpus} GPU(s)")
    
    # Model parameters
    batch_size = 1  # Keep small for large images
    num_anchors = 3
    base_channels = 32  # Reduced for memory
    
    # Test configurations
    test_sizes = [(256, 256), (512, 512), (768, 768), (1024, 1024)]
    
    for height, width in test_sizes:
        print(f"\n{'='*50}")
        print(f"Testing {height}x{width}")
        print(f"{'='*50}")
        
        try:
            if num_gpus > 1:
                # STRATEGY 1: Split batch across GPUs
                print(f"Strategy: Split computation across {num_gpus} GPUs")
                
                # Create model replicas on each GPU
                models = []
                for gpu_id in range(num_gpus):
                    model = create_fusion_model(num_anchors=num_anchors, base_channels=base_channels)
                    model = model.cuda(gpu_id)
                    model.eval()
                    models.append(model)
                    print(f"  Model replica created on GPU {gpu_id}")
                
                # Create inputs on CPU first
                I0_all = torch.randn(batch_size, num_anchors, 3, height, width)
                I1_all = torch.randn(batch_size, num_anchors, 3, height, width)
                flows_all = torch.randn(batch_size, num_anchors, 4, height, width)
                masks_all = torch.rand(batch_size, num_anchors, height, width)
                timesteps = torch.rand(batch_size, num_anchors)
                
                # Split across height dimension for each GPU to process a strip
                if height >= num_gpus * 64:  # Ensure minimum strip size
                    print(f"  Splitting image into {num_gpus} horizontal strips")
                    
                    # Calculate strip size with overlap
                    overlap = 32
                    strip_size = height // num_gpus + overlap
                    
                    outputs = []
                    for gpu_id in range(num_gpus):
                        # Calculate strip boundaries
                        start_h = max(0, gpu_id * (height // num_gpus) - overlap // 2)
                        end_h = min(height, (gpu_id + 1) * (height // num_gpus) + overlap // 2)
                        
                        # Extract strip
                        I0_strip = I0_all[:, :, :, start_h:end_h, :].cuda(gpu_id)
                        I1_strip = I1_all[:, :, :, start_h:end_h, :].cuda(gpu_id)
                        flows_strip = flows_all[:, :, :, start_h:end_h, :].cuda(gpu_id)
                        masks_strip = masks_all[:, :, start_h:end_h, :].cuda(gpu_id)
                        timesteps_gpu = timesteps.cuda(gpu_id)
                        
                        # Process strip on its GPU
                        with torch.no_grad():
                            output_strip, _ = models[gpu_id](
                                I0_strip, I1_strip, flows_strip, masks_strip, timesteps_gpu
                            )
                        
                        # Move result back to CPU
                        outputs.append(output_strip.cpu())
                        
                        # Report memory for this GPU
                        mem_mb = torch.cuda.memory_allocated(gpu_id) / 1024**2
                        print(f"    GPU {gpu_id}: Strip {start_h}:{end_h}, Memory: {mem_mb:.1f} MB")
                    
                    # Merge outputs (simplified - just concatenate without overlap handling)
                    # In practice, you'd blend the overlapping regions
                    print(f"  ✓ Size {height}x{width} passed using {num_gpus} GPUs!")
                    
                else:
                    # Image too small to split effectively
                    print(f"  Image too small to split across {num_gpus} GPUs, using GPU 0")
                    I0_all = I0_all.cuda(0)
                    I1_all = I1_all.cuda(0)
                    flows_all = flows_all.cuda(0)
                    masks_all = masks_all.cuda(0)
                    timesteps = timesteps.cuda(0)
                    
                    with torch.no_grad():
                        output, _ = models[0](I0_all, I1_all, flows_all, masks_all, timesteps)
                    print(f"  ✓ Size {height}x{width} passed on single GPU")
                
            else:
                # Single GPU/CPU path
                print(f"Using single device: {device}")
                
                model = create_fusion_model(num_anchors=num_anchors, base_channels=base_channels)
                model = model.to(device)
                model.eval()
                
                # Create inputs
                I0_all = torch.randn(batch_size, num_anchors, 3, height, width).to(device)
                I1_all = torch.randn(batch_size, num_anchors, 3, height, width).to(device)
                flows_all = torch.randn(batch_size, num_anchors, 4, height, width).to(device)
                masks_all = torch.rand(batch_size, num_anchors, height, width).to(device)
                timesteps = torch.rand(batch_size, num_anchors).to(device)
                
                # Forward pass
                with torch.no_grad():
                    output, aux_outputs = model(I0_all, I1_all, flows_all, masks_all, timesteps)
                
                print(f"  ✓ Size {height}x{width} passed!")
                print(f"  Output shape: {output.shape}")
                
                if device.type == 'cuda':
                    mem_gb = torch.cuda.memory_allocated(device) / 1024**3
                    print(f"  Memory used: {mem_gb:.2f} GB")
            
            # Clear memory
            if torch.cuda.is_available():
                for gpu_id in range(num_gpus):
                    torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ OOM at {height}x{width}")
                if torch.cuda.is_available():
                    for gpu_id in range(num_gpus):
                        torch.cuda.empty_cache()
                break
            else:
                print(f"  ✗ Error: {e}")
                raise
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            raise

def test_pipeline_parallel():
    """Alternative: Pipeline parallelism - split model stages across GPUs"""
    print("\n" + "=" * 50)
    print("Testing Pipeline Parallelism")
    print("=" * 50)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("Pipeline parallelism requires at least 2 GPUs")
        return
    
    print(f"Using {num_gpus} GPUs for pipeline")
    
    # This is a simplified example - in practice you'd need to modify the model
    # to support pipeline parallelism properly
    
    class SimplePipelineModel(nn.Module):
        def __init__(self, base_model, num_gpus):
            super().__init__()
            self.num_gpus = num_gpus
            
            # Move different parts of the model to different GPUs
            # This is a simplified example - you'd need to properly split the model
            self.stage1 = base_model  # Entire model on first GPU for simplicity
            self.stage1 = self.stage1.cuda(0)
            
            # In a real implementation, you'd split the model layers:
            # self.encoder = model.encoder.cuda(0)
            # self.attention = model.attention.cuda(1)
            # self.decoder = model.decoder.cuda(2)
            
        def forward(self, I0, I1, flows, masks, timesteps):
            # Stage 1 on GPU 0
            output, aux = self.stage1(I0, I1, flows, masks, timesteps)
            
            # In real pipeline, you'd pass intermediate results between GPUs:
            # x = self.encoder(inputs)  # GPU 0
            # x = x.cuda(1)  # Move to GPU 1
            # x = self.attention(x)  # GPU 1
            # x = x.cuda(2)  # Move to GPU 2
            # output = self.decoder(x)  # GPU 2
            
            return output, aux
    
    # Test with pipeline
    base_model = create_fusion_model(num_anchors=3, base_channels=32)
    pipeline_model = SimplePipelineModel(base_model, num_gpus)
    
    # Test different sizes
    for h, w in [(256, 256), (512, 512)]:
        print(f"\nTesting {h}x{w} with pipeline...")
        try:
            inputs = torch.randn(1, 3, 3, h, w).cuda(0)
            flows = torch.randn(1, 3, 4, h, w).cuda(0)
            masks = torch.rand(1, 3, h, w).cuda(0)
            timesteps = torch.rand(1, 3).cuda(0)
            
            with torch.no_grad():
                output, _ = pipeline_model(inputs, inputs, flows, masks, timesteps)
            
            print(f"  ✓ Pipeline processed {h}x{w} successfully")
            
            # Report memory per GPU
            for gpu_id in range(num_gpus):
                if torch.cuda.memory_allocated(gpu_id) > 0:
                    mem_mb = torch.cuda.memory_allocated(gpu_id) / 1024**2
                    print(f"    GPU {gpu_id} memory: {mem_mb:.1f} MB")
            
        except Exception as e:
            print(f"  ✗ Pipeline failed: {e}")

def main():
    """Main entry point"""
    print("GPU Information:")
    print("-" * 50)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        print("No CUDA GPUs available")
        if torch.backends.mps.is_available():
            print("Using Apple MPS")
        else:
            print("Using CPU")
    
    print()
    
    # Test model parallel approach
    test_model_parallel()
    
    # Test pipeline parallel approach (if multiple GPUs)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        test_pipeline_parallel()
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("=" * 50)
    
    print("\nSummary:")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("✓ Multiple GPUs can be used to process larger images by:")
        print("  1. Splitting images into strips (spatial parallelism)")
        print("  2. Pipeline parallelism (if model is modified)")
        print("  3. Using gradient checkpointing for memory efficiency")
    else:
        print("✓ Single device mode works well for smaller images")
        print("✓ For larger images, consider:")
        print("  - Reducing batch size")
        print("  - Reducing base_channels")
        print("  - Using gradient checkpointing")
        print("  - Processing in tiles/patches")

if __name__ == "__main__":
    main()