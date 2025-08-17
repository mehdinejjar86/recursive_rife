#!/usr/bin/env python3
"""Test script to verify model dimensions and functionality"""

import torch
import torch.nn as nn
from fusion_model import create_fusion_model, MultiAnchorFusionModel

def test_model_forward():
    """Test the model forward pass with correct dimensions"""
    print("=" * 50)
    print("Testing Multi-Anchor Fusion Model")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                          'cpu')
    print(f"Device: {device}\n")
    
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
    model = create_fusion_model(num_anchors=num_anchors, base_channels=base_channels).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Create dummy inputs
    print("Creating test inputs...")
    I0_all = torch.randn(batch_size, num_anchors, 3, height, width).to(device)
    I1_all = torch.randn(batch_size, num_anchors, 3, height, width).to(device)
    flows_all = torch.randn(batch_size, num_anchors, 4, height, width).to(device)
    masks_all = torch.rand(batch_size, num_anchors, height, width).to(device)
    timesteps = torch.rand(batch_size, num_anchors).to(device)
    
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
        if torch.cuda.is_available() and device.type == 'cuda':
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_sizes():
    """Test model with different input sizes"""
    print("\n" + "=" * 50)
    print("Testing Different Input Sizes")
    print("=" * 50)
    
    device = torch.device('cpu')  # Use CPU for size testing
    model = create_fusion_model(num_anchors=3, base_channels=32).to(device)
    model.eval()
    
    test_sizes = [(64, 64), (128, 128), (256, 256), ()(512, 512), (1024, 1024), (2048, 2048)]
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
        except Exception as e:
            print(f"  ❌ Size {h}x{w} failed: {e}")

def test_gradient_flow():
    """Test that gradients flow properly through the model"""
    print("\n" + "=" * 50)
    print("Testing Gradient Flow")
    print("=" * 50)
    
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

if __name__ == "__main__":
    # Run tests
    success = test_model_forward()
    
    if success:
        test_different_sizes()
        test_gradient_flow()
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("=" * 50)