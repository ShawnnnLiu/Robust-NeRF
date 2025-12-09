# Joint NeRF and Camera Pose Optimization

This document describes the **core feature** of this project: joint optimization of NeRF scene representation and camera extrinsics (rotation and translation).

## Overview

Traditional NeRF requires accurate camera poses. This implementation allows NeRF to **learn and refine** camera poses during training, making it robust to noisy or imperfect camera calibration.

### Key Features

✅ **Joint Optimization**: Simultaneously optimize scene representation (NeRF) and camera parameters  
✅ **Flexible Parameterization**: SE(3) with axis-angle rotations (no gimbal lock)  
✅ **Configurable**: Learn rotation only, translation only, or both  
✅ **Robust Initialization**: Works with clean or noisy camera pose initialization  
✅ **Staged Training**: Optional delayed pose optimization for stability  
✅ **Comprehensive Tracking**: Monitor pose refinement throughout training

## Architecture

### Core Components

1. **`CameraPoseParameters`** (`train_pose_opt.py`)
   - Learnable camera parameters using SE(3) representation
   - Axis-angle for rotations (avoids gimbal lock)
   - Stores deltas from initial poses for stability

2. **`PixelDataset` & `PixelSampler`** (`data_pose_opt.py`)
   - Stores pixel coordinates instead of precomputed rays
   - Regenerates rays from updated poses during training
   - Essential for proper gradient flow to camera parameters

3. **`train_with_pose_optimization()`** (`train_pose_opt.py`)
   - Main training loop with joint optimization
   - Separate optimizers and learning rates for NeRF and poses
   - Gradient clipping for stable pose updates

## Usage

### Quick Start

```bash
# Basic joint optimization with noisy initialization
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode noisy \
    --rotation_noise 2.0 \
    --translation_noise_pct 1.0 \
    --num_iters 50000

# Clean initialization (verify system maintains performance)
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode clean \
    --num_iters 30000

# Optimize only rotations
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode noisy \
    --rotation_noise 2.0 \
    --no_learn_translation

# Optimize only translations
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode noisy \
    --translation_noise_pct 1.0 \
    --no_learn_rotation
```

### Command-Line Arguments

#### Initialization Mode
- `--init_mode {clean,noisy}`: How to initialize camera poses
  - `clean`: Use ground truth poses (verifies system doesn't degrade performance)
  - `noisy`: Add noise to poses (demonstrates robustness)

#### Pose Optimization Settings
- `--pose_lr FLOAT`: Learning rate for camera poses (default: 1e-4)
- `--pose_opt_delay INT`: Start pose optimization after N iterations (default: 1000)
- `--no_learn_rotation`: Freeze rotation parameters (translation only)
- `--no_learn_translation`: Freeze translation parameters (rotation only)

#### Noise Configuration (for noisy initialization)
- `--rotation_noise FLOAT`: Rotation noise std in degrees
- `--translation_noise FLOAT`: Translation noise std in scene units
- `--translation_noise_pct FLOAT`: Translation noise as % of camera distance
- `--noise_seed INT`: Random seed for reproducible noise

#### Standard NeRF Arguments
- `--scene STR`: Scene name (e.g., lego, chair, drums)
- `--batch_size INT`: Rays per batch (default: 1024)
- `--num_iters INT`: Training iterations (default: 50000)
- `--lr FLOAT`: NeRF learning rate (default: 5e-4)
- `--device {cuda,cpu}`: Device to use

## Examples

### Example 1: Clean Baseline

Verify the system maintains performance when initialized with ground truth:

```bash
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode clean \
    --num_iters 30000 \
    --val_every 2500
```

**Expected**: Poses should stay near ground truth, rendering quality should match baseline NeRF.

### Example 2: Rotation Noise Recovery

Recover from imperfect camera calibration (rotation errors):

```bash
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode noisy \
    --rotation_noise 2.0 \
    --num_iters 40000 \
    --pose_lr 1e-4
```

**Expected**: Rotation errors should decrease significantly during training.

### Example 3: Full Joint Optimization

Most realistic scenario with both rotation and translation noise:

```bash
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode noisy \
    --rotation_noise 2.0 \
    --translation_noise_pct 1.0 \
    --num_iters 50000 \
    --pose_opt_delay 1000
```

**Expected**: Both rotation and translation errors should decrease, final rendering quality should approach clean baseline.

### Example 4: Extreme Noise

Test system limits with severe noise:

```bash
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode noisy \
    --rotation_noise 5.0 \
    --translation_noise_pct 2.0 \
    --num_iters 60000 \
    --pose_lr 2e-4 \
    --pose_opt_delay 500
```

**Expected**: Slower convergence, but system should still refine poses and produce reasonable results.

### Example 5: Delayed Optimization

Two-stage training for increased stability:

```bash
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode noisy \
    --rotation_noise 2.0 \
    --translation_noise_pct 1.0 \
    --num_iters 50000 \
    --pose_opt_delay 10000
```

**Expected**: NeRF learns coarse scene first, then refines poses for fine details.

## Output Structure

Training produces the following outputs in `outputs/{experiment_name}/`:

```
outputs/lego_poseopt_noisyinit_rot2.0deg_trans1.0pct_20251208_120000/
├── checkpoint_*.pt              # Regular checkpoints
├── checkpoint_best.pt           # Best validation PSNR checkpoint
├── checkpoint_latest.pt         # Latest checkpoint
├── final_poses.pt              # Final pose comparison
│   ├── initial_poses           # Starting poses (noisy)
│   ├── optimized_poses         # Learned poses
│   ├── ground_truth_poses      # GT for comparison
│   └── pose_errors             # Final error statistics
├── experiment_config.json      # Full experiment configuration
├── images/                     # Validation renderings
├── logs/                       # Training logs and TensorBoard
└── config.json                 # Model/render configuration
```

### Checkpoint Contents

Each checkpoint includes:
- NeRF model weights (coarse + fine)
- Camera pose parameters (deltas and initial poses)
- Optimizers state
- Current pose errors (rotation and translation)
- Validation metrics

### Analyzing Results

```python
import torch

# Load final poses
poses = torch.load('outputs/.../final_poses.pt')

initial_poses = poses['initial_poses']
optimized_poses = poses['optimized_poses']
gt_poses = poses['ground_truth_poses']
errors = poses['pose_errors']

print(f"Final rotation error: {errors['rotation_error_mean']:.3f}° ± {errors['rotation_error_std']:.3f}°")
print(f"Final translation error: {errors['translation_error_mean']:.4f} ± {errors['translation_error_std']:.4f}")
```

## Technical Details

### SE(3) Parameterization

Camera poses are parameterized using SE(3) with axis-angle rotations:

```
R_optimized = exp(ω) ⊗ R_initial
t_optimized = t_initial + δt

where:
  ω ∈ ℝ³ is the axis-angle rotation delta
  δt ∈ ℝ³ is the translation delta
  exp(ω) is the exponential map (Rodrigues' formula)
```

This approach:
- Avoids gimbal lock (no Euler angles)
- Provides minimal parameterization (3 DOF for rotation)
- Allows unbounded optimization (no constraints needed)
- Maintains numerical stability

### Gradient Flow

The key to successful joint optimization is proper gradient flow:

1. **Pixel Sampling**: Sample random pixels (not precomputed rays)
2. **Ray Generation**: Generate rays from current (learnable) poses
3. **Rendering**: Render NeRF along rays
4. **Loss Computation**: MSE between rendered and target colors
5. **Backpropagation**: Gradients flow to both NeRF and pose parameters

```python
# Simplified gradient flow
poses = camera_params.get_poses()          # Learnable poses
rays_o, rays_d = generate_rays(poses)      # Differentiable w.r.t. poses
rgb = render_nerf(rays_o, rays_d)          # Differentiable w.r.t. NeRF + rays
loss = mse(rgb, target)                    # Differentiable w.r.t. rgb
loss.backward()                            # Backprop to NeRF AND poses
```

### Learning Rate Scheduling

Different learning rates are used for different components:

- **NeRF**: 5e-4 (standard, decays exponentially)
- **Camera Poses**: 1e-4 to 2e-4 (lower, more conservative)
- **Gradient Clipping**: 
  - NeRF: max norm 1.0
  - Poses: max norm 0.1 (more aggressive clipping for stability)

### Training Strategies

1. **Warm-up**: Train NeRF for 500-1000 iterations before optimizing poses
   - Allows NeRF to learn coarse scene structure
   - Provides better gradients for pose refinement

2. **Separate Optimizers**: Use separate Adam optimizers for NeRF and poses
   - Different learning rates
   - Independent momentum

3. **Staged Optimization**: Optionally freeze poses early, unfreeze later
   - More stable for severe noise
   - Prevents early instability

## Comparison with Baselines

### Baseline 1: Fixed Noisy Poses

Train NeRF with fixed noisy poses (no optimization):

```bash
python -m noisy_src.train \
    --scene lego \
    --rotation_noise 2.0 \
    --translation_noise_pct 1.0 \
    --num_iters 50000
```

**Expected**: Poor rendering quality, blurry images, low PSNR.

### Baseline 2: Clean Poses

Train NeRF with ground truth poses:

```bash
python -m noisy_src.train \
    --scene lego \
    --num_iters 50000
```

**Expected**: Best possible quality for reference.

### Joint Optimization (This Method)

Train with joint optimization:

```bash
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode noisy \
    --rotation_noise 2.0 \
    --translation_noise_pct 1.0 \
    --num_iters 50000
```

**Expected**: Quality approaching clean baseline, refined poses.

## Limitations

1. **Computational Cost**: ~10-20% slower than fixed-pose training due to ray regeneration
2. **Memory**: Slightly higher memory usage for storing pixel coordinates
3. **Convergence**: May require more iterations for severe noise (>5° rotation)
4. **Ambiguity**: Cannot resolve scale ambiguity (inherent to structure-from-motion)

## Best Practices

1. **Start Conservative**: Begin with small noise levels (1-2°) to verify system
2. **Monitor Convergence**: Check pose errors during training (logged every val_every iterations)
3. **Adjust Learning Rates**: Increase pose_lr for severe noise, decrease for instability
4. **Use Warmup**: Always use pose_opt_delay > 0 for stability
5. **Validate on Multiple Scenes**: Different scenes have different sensitivities to pose noise

## Related Work

This implementation is inspired by:
- **BARF** (Bundle-Adjusting Neural Radiance Fields): Coarse-to-fine pose optimization
- **NeRF--** (NeRF Minus Minus): Joint pose and NeRF optimization
- **Self-Calibrating Neural Radiance Fields**: Learning camera intrinsics and extrinsics

Our approach focuses on **simplicity** and **clarity** while maintaining effectiveness.

## Future Enhancements

Potential improvements:
- [ ] Per-frame adaptive learning rates
- [ ] Coarse-to-fine pose optimization (BARF-style)
- [ ] Support for optimizing camera intrinsics (focal length, distortion)
- [ ] Multi-scale training for faster convergence
- [ ] Uncertainty estimation for poses
- [ ] Support for sequential pose refinement (SLAM-like)

## Citation

If you use this code, please cite:

```bibtex
@software{robust_nerf_2024,
  title = {Robust NeRF: Joint Optimization with Camera Pose Refinement},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Robust-NeRF}
}
```

## Questions?

For issues or questions:
1. Check the examples in `scripts/train_pose_optimization.py`
2. Review the documentation in this file
3. Open an issue on GitHub

Happy training! 🚀

