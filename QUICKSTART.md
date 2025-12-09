# Quick Start Guide: Joint NeRF and Pose Optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Robust-NeRF.git
cd Robust-NeRF

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NeRF Synthetic dataset
# From: https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4
# Extract to: data/raw/nerf_synthetic/
```

## Quick Commands

### 1. Baseline Training (Clean Poses)

```bash
python -m noisy_src.train \
    --scene lego \
    --num_iters 50000 \
    --device cuda
```

⏱️ Time: ~2-3 hours on GPU  
📊 Expected PSNR: 30-32 dB

---

### 2. Joint Optimization (Recommended)

```bash
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode noisy \
    --rotation_noise 2.0 \
    --translation_noise_pct 1.0 \
    --num_iters 50000 \
    --pose_lr 1e-4 \
    --device cuda
```

⏱️ Time: ~3-4 hours on GPU  
📊 Expected: Pose errors decrease, PSNR approaches baseline

---

### 3. Fixed Noisy (Show Degradation)

```bash
python -m noisy_src.train \
    --scene lego \
    --rotation_noise 2.0 \
    --translation_noise_pct 1.0 \
    --num_iters 50000 \
    --device cuda
```

⏱️ Time: ~2-3 hours on GPU  
📊 Expected: Low PSNR (~20-25 dB), blurry images

---

## Common Options

### Scenes
- `--scene lego` (default)
- `--scene chair`
- `--scene drums`
- `--scene ship`

### Noise Levels
- Mild: `--rotation_noise 1.0 --translation_noise_pct 0.5`
- Moderate: `--rotation_noise 2.0 --translation_noise_pct 1.0` ⭐
- Severe: `--rotation_noise 5.0 --translation_noise_pct 2.0`

### Training Length
- Quick test: `--num_iters 10000` (10 mins)
- Short: `--num_iters 30000` (1 hour)
- Standard: `--num_iters 50000` (2-3 hours) ⭐
- Full: `--num_iters 200000` (8-12 hours)

### Pose Optimization
- Both: Default (rotation + translation)
- Rotation only: `--no_learn_translation`
- Translation only: `--no_learn_rotation`

---

## Output Structure

```
outputs/lego_poseopt_noisyinit_rot2.0deg_trans1.0pct_20251208_120000/
├── checkpoint_best.pt              # Best model
├── final_poses.pt                  # Initial vs optimized vs GT poses
├── experiment_config.json          # Configuration
├── images/                         # Validation images
│   ├── val_0_comparison_*.png
│   └── val_0_depth_*.png
└── logs/
    ├── train_metrics.csv          # Training metrics
    ├── val_metrics.csv            # Validation metrics
    └── tensorboard/               # TensorBoard logs
```

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir outputs/
```

Then open: http://localhost:6006

### Check Pose Errors

Look for these lines during validation:

```
Pose Refinement:
  Rotation error:    0.234° ± 0.156° (max: 0.512°)
  Translation error: 0.0023 ± 0.0012 (max: 0.0045)
```

### Check Rendering Quality

```
Rendering Quality:
  PSNR:  31.25 dB
  SSIM:  0.9621
```

---

## Comparing Results

### Load and Compare Poses

```python
import torch
import numpy as np

# Load results
poses = torch.load('outputs/.../final_poses.pt')

initial = poses['initial_poses']
optimized = poses['optimized_poses']
gt = poses['ground_truth_poses']

# Compute improvement
errors = poses['pose_errors']
print(f"Final rotation error: {errors['rotation_error_mean']:.3f}°")
print(f"Final translation error: {errors['translation_error_mean']:.4f}")
```

### Load Metrics

```python
import pandas as pd

# Load training metrics
train_df = pd.read_csv('outputs/.../logs/train_metrics.csv')
val_df = pd.read_csv('outputs/.../logs/val_metrics.csv')

# Plot learning curves
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_df['iteration'], train_df['psnr'])
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('Training PSNR')

plt.subplot(1, 2, 2)
plt.plot(val_df['iteration'], val_df['psnr'])
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('Validation PSNR')
plt.show()
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
--batch_size 512  # Default is 1024
```

Or reduce image resolution:
```bash
--img_scale 0.25  # Default is 0.5 (half resolution)
```

### Pose Optimization Unstable

- Reduce pose learning rate: `--pose_lr 5e-5`
- Increase delay: `--pose_opt_delay 2000`
- Use two-stage training (delay until NeRF converges)

### Poor Convergence

- Increase iterations: `--num_iters 100000`
- Check pose errors are decreasing (in validation logs)
- Try lower noise levels first

### Slow Training

- Use smaller resolution: `--img_scale 0.25`
- Reduce samples: `--num_samples 32 --num_samples_fine 64`
- Use fewer iterations for testing: `--num_iters 10000`

---

## Next Steps

1. **Run baseline**: Verify installation with clean poses
2. **Run joint optimization**: Test with moderate noise
3. **Compare results**: Check pose refinement and rendering quality
4. **Experiment**: Try different scenes, noise levels, optimization strategies

📖 **Detailed docs**: See [POSE_OPTIMIZATION.md](POSE_OPTIMIZATION.md)  
🔧 **Examples**: See `scripts/train_pose_optimization.py`  
💬 **Questions**: Open an issue on GitHub

Happy training! 🚀

