"""
Quick test script to verify the NeRF baseline implementation.

Run with: python -m noisy_src.test_baseline
"""

import torch
import sys
from pathlib import Path


def test_model():
    """Test NeRF model creation and forward pass."""
    from .model import NeRF, create_nerf, PositionalEncoding
    from .config import ModelConfig
    
    print("Testing model...")
    
    # Test positional encoding
    pe = PositionalEncoding(num_freqs=10, include_input=True)
    x = torch.randn(100, 3)
    encoded = pe(x)
    expected_dim = 3 * (1 + 2 * 10)  # input + sin/cos for each freq
    assert encoded.shape == (100, expected_dim), f"Expected {(100, expected_dim)}, got {encoded.shape}"
    print(f"  ✓ Positional encoding: {x.shape} -> {encoded.shape}")
    
    # Test NeRF model
    config = ModelConfig()
    model = NeRF(config)
    
    pts = torch.randn(1024, 3)
    dirs = torch.randn(1024, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    
    rgb, sigma = model(pts, dirs)
    assert rgb.shape == (1024, 3), f"RGB shape mismatch: {rgb.shape}"
    assert sigma.shape == (1024, 1), f"Sigma shape mismatch: {sigma.shape}"
    assert rgb.min() >= 0 and rgb.max() <= 1, "RGB should be in [0, 1]"
    assert sigma.min() >= 0, "Sigma should be non-negative"
    print(f"  ✓ NeRF forward: pts {pts.shape}, dirs {dirs.shape} -> rgb {rgb.shape}, sigma {sigma.shape}")
    
    # Test coarse/fine creation
    coarse, fine = create_nerf(config)
    assert coarse is not None
    assert fine is not None
    print("  ✓ Coarse/fine model creation")
    
    print("Model tests passed!\n")


def test_rays():
    """Test ray generation and sampling."""
    from .rays import get_ray_directions, get_rays, sample_along_rays, sample_hierarchical
    
    print("Testing ray utilities...")
    
    H, W, focal = 100, 100, 50.0
    
    # Test ray directions
    directions = get_ray_directions(H, W, focal)
    assert directions.shape == (H, W, 3), f"Direction shape: {directions.shape}"
    print(f"  ✓ Ray directions: {directions.shape}")
    
    # Test ray generation with camera pose
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([0, 0, 4])  # Camera at z=4
    
    rays_o, rays_d = get_rays(directions, c2w)
    assert rays_o.shape == (H, W, 3)
    assert rays_d.shape == (H, W, 3)
    print(f"  ✓ Ray generation: origins {rays_o.shape}, directions {rays_d.shape}")
    
    # Test sampling
    rays_o_flat = rays_o.reshape(-1, 3)[:100]
    rays_d_flat = rays_d.reshape(-1, 3)[:100]
    
    pts, z_vals = sample_along_rays(
        rays_o_flat, rays_d_flat,
        near=2.0, far=6.0,
        num_samples=64,
        perturb=True
    )
    assert pts.shape == (100, 64, 3)
    assert z_vals.shape == (100, 64)
    print(f"  ✓ Stratified sampling: {pts.shape}")
    
    # Test hierarchical sampling
    weights = torch.rand(100, 64)
    pts_fine, z_vals_fine = sample_hierarchical(
        rays_o_flat, rays_d_flat,
        z_vals, weights,
        num_samples_fine=128,
    )
    assert pts_fine.shape == (100, 64 + 128, 3)
    print(f"  ✓ Hierarchical sampling: {pts_fine.shape}")
    
    print("Ray tests passed!\n")


def test_rendering():
    """Test volume rendering."""
    from .rendering import raw2outputs, NeRFRenderer
    from .model import create_nerf
    from .config import ModelConfig, RenderConfig
    
    print("Testing rendering...")
    
    # Test raw2outputs
    N_rays = 100
    N_samples = 64
    
    rgb = torch.rand(N_rays, N_samples, 3)
    sigma = torch.rand(N_rays, N_samples, 1) * 10
    z_vals = torch.linspace(2, 6, N_samples).unsqueeze(0).expand(N_rays, -1)
    rays_d = torch.randn(N_rays, 3)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    
    outputs = raw2outputs(rgb, sigma, z_vals, rays_d)
    
    assert outputs["rgb_map"].shape == (N_rays, 3)
    assert outputs["depth_map"].shape == (N_rays,)
    assert outputs["acc_map"].shape == (N_rays,)
    assert outputs["weights"].shape == (N_rays, N_samples)
    print(f"  ✓ Volume rendering: {outputs['rgb_map'].shape}")
    
    # Test full renderer
    config = ModelConfig()
    render_config = RenderConfig(num_samples=32, num_samples_fine=64)
    
    coarse, fine = create_nerf(config)
    renderer = NeRFRenderer(coarse, fine, render_config)
    
    rays_o = torch.zeros(50, 3)
    rays_o[:, 2] = 4.0
    rays_d = torch.randn(50, 3)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    
    with torch.no_grad():
        results = renderer(rays_o, rays_d, chunk_size=25, is_train=False)
    
    assert "rgb_coarse" in results
    assert "rgb_fine" in results
    assert results["rgb_fine"].shape == (50, 3)
    print(f"  ✓ NeRF renderer: {results['rgb_fine'].shape}")
    
    print("Rendering tests passed!\n")


def test_data_loading():
    """Test data loading (if data exists)."""
    from .data import load_blender_data
    from .config import DataConfig
    
    print("Testing data loading...")
    
    # Find data directory
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw"
    
    if not data_root.exists():
        print("  ⚠ Data directory not found, skipping data tests")
        return
    
    # Try to load a scene
    try:
        data = load_blender_data(
            data_root=data_root,
            scene_name="chair",  # Using chair as it's in your project
            split="train",
            img_scale=0.25,  # Use lower resolution for testing
            device="cpu",
        )
        
        print(f"  ✓ Loaded scene: {data.images.shape[0]} images")
        print(f"    Resolution: {data.H}x{data.W}")
        print(f"    Focal: {data.focal:.2f}")
        print(f"    Image range: [{data.images.min():.2f}, {data.images.max():.2f}]")
        
    except FileNotFoundError as e:
        print(f"  ⚠ Scene not found: {e}")
        return
    
    print("Data loading tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("NeRF Baseline Implementation Tests")
    print("=" * 60 + "\n")
    
    try:
        test_model()
        test_rays()
        test_rendering()
        test_data_loading()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nTo train the baseline NeRF:")
        print("  python -m noisy_src.train --scene chair --exp_name chair_baseline")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()





