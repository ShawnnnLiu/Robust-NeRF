## 🎯 Robust NeRF: Joint Optimization with Camera Pose Refinement

A complete implementation of **NeRF with joint camera pose optimization** for noise-robust 3D reconstruction. This project demonstrates how Neural Radiance Fields can **learn and refine** camera extrinsics during training, making them robust to noisy or imperfect camera calibration.

### ✨ Key Features

- ✅ **Complete NeRF Implementation**: Full vanilla NeRF with hierarchical sampling
- ✅ **Joint Pose Optimization**: Simultaneously optimize scene and camera parameters
- ✅ **SE(3) Parameterization**: Axis-angle rotations for stable optimization
- ✅ **Flexible Configuration**: Learn rotation only, translation only, or both
- ✅ **Comprehensive Logging**: TensorBoard, metrics tracking, pose error monitoring
- ✅ **Multiple Training Modes**: Clean baseline, noisy fixed, and joint optimization

### 🚀 Quick Start

Train NeRF with joint camera pose optimization:

```bash
# Basic joint optimization with noisy poses
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode noisy \
    --rotation_noise 2.0 \
    --translation_noise_pct 1.0 \
    --num_iters 50000

# Clean baseline for comparison
python -m noisy_src.train \
    --scene lego \
    --num_iters 50000
```

See **[POSE_OPTIMIZATION.md](POSE_OPTIMIZATION.md)** for detailed documentation and examples.

---

### 📁 Repository Structure

```
Robust-NeRF/
├── noisy_src/                          # Main source code
│   ├── config.py                       # Configuration dataclasses
│   ├── model.py                        # NeRF MLP architecture
│   ├── rendering.py                    # Volume rendering
│   ├── rays.py                         # Ray generation and sampling
│   ├── data.py                         # Data loading utilities
│   ├── noise.py                        # Noise injection utilities
│   ├── train.py                        # Standard NeRF training
│   ├── train_pose_opt.py              # 🌟 Joint pose optimization training
│   ├── data_pose_opt.py               # Pixel-based data loader for pose opt
│   ├── metrics.py                      # PSNR, SSIM, LPIPS metrics
│   ├── logger.py                       # Experiment logging
│   └── utils.py                        # Utility functions
├── scripts/
│   ├── inject_noise.py                 # Noise injection tools
│   └── train_pose_optimization.py     # Example training scripts
├── notebooks/
│   ├── explore_data.ipynb             # Data exploration
│   └── visualize_noise_effects.ipynb  # Noise visualization
├── outputs/                            # Training outputs
├── data/raw/                           # Dataset location
├── POSE_OPTIMIZATION.md               # 📖 Detailed documentation
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

---

### Dataset: NeRF Synthetic Blender (lego)

This project uses the **NeRF Synthetic Blender dataset**, specifically the **lego** scene.

For each scene, the dataset typically provides:

- **RGB images** (often split into `train/`, `val/`, `test/`).
- **Camera transforms**:
  - A JSON file (e.g., `transforms.json` or `transforms_train.json`) containing **camera-to-world** transform matrices.
  - Approximate **field-of-view (FOV)** information.

You can download the official NeRF synthetic data bundle (which includes `lego`) from:

- **Google Drive (NeRF data)**: `https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4`

Download **`nerf_synthetic.zip`** from that folder and unzip it into `data/raw/`, so you end up with a structure like:

```text
data/raw/nerf_synthetic/
  lego/
    train/
    val/
    test/
    transforms_train.json
    transforms_val.json
    transforms_test.json
  chair/
  drums/
  ...
```

**Note:** The notebooks and training scripts will automatically look for data in `data/raw/nerf_synthetic/lego/` by default.

For convenience, you can optionally create a symlink so the project can also refer to `data/raw/lego`:

```bash
cd data/raw
ln -s nerf_synthetic/lego lego
cd ../..
```

The code has fallback logic to check both locations automatically.

---

### Getting Started

#### 1. Set up the Python environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Download and unpack the NeRF synthetic dataset

1. Go to the NeRF data Google Drive folder:  
   `https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4`
2. Download **`nerf_synthetic.zip`**.
3. From the repository root, create the data directory (if needed) and unzip:

   ```bash
   mkdir -p data/raw
   unzip /path/to/nerf_synthetic.zip -d data/raw
   ```

4. (Optional) Create a symlink for the lego scene:

   ```bash
   cd data/raw
   ln -s nerf_synthetic/lego lego
   cd ../..
   ```

The notebooks and scripts will look for the lego scene in `data/raw/nerf_synthetic/lego/` by default, but they also have fallback logic to check `data/raw/lego/` if the symlink is created.

---

### Visualizing the Dataset (Notebooks)

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then open:

- **`notebooks/explore_data.ipynb`**
  - Displays a few sample RGB frames from the lego scene.
  - Visualizes the distribution of camera poses in 3D.
- **`notebooks/visualize_noise_effects.ipynb`**
  - Load camera transforms.
  - Apply simple Gaussian noise to extrinsics.
  - Visualize original vs noisy camera trajectories.

---

### 🧪 Training Modes

#### 1. Baseline NeRF (Clean Poses)

Train standard NeRF with ground truth camera poses:

```bash
python -m noisy_src.train --scene lego --num_iters 200000
```

**Use case**: Establish performance baseline.

#### 2. Fixed Noisy Poses

Train NeRF with noisy poses (fixed, not optimized):

```bash
python -m noisy_src.train \
    --scene lego \
    --rotation_noise 2.0 \
    --translation_noise_pct 1.0 \
    --num_iters 200000
```

**Use case**: Demonstrate degradation from noisy poses.

#### 3. Joint Pose Optimization ⭐

Train NeRF while simultaneously refining camera poses:

```bash
python -m noisy_src.train_pose_opt \
    --scene lego \
    --init_mode noisy \
    --rotation_noise 2.0 \
    --translation_noise_pct 1.0 \
    --num_iters 50000 \
    --pose_lr 1e-4
```

**Use case**: Recover from noisy initialization, achieve robust reconstruction.

---

### 📊 Key Results

The joint optimization approach can:

- ✅ **Refine poses** from noisy initialization (2-5° rotation, 1-2% translation)
- ✅ **Maintain quality** approaching clean baseline
- ✅ **Converge stably** with proper learning rate scheduling
- ✅ **Scale** to different scenes and noise levels

See example outputs in `outputs/` directory after training.

---

### 🛠️ Implementation Highlights

#### Camera Pose Parameterization

Uses **SE(3)** with axis-angle rotations:
- No gimbal lock issues
- Minimal parameterization (3 DOF rotation + 3 DOF translation)
- Stable gradient flow

```python
R_optimized = exp(ω) ⊗ R_initial
t_optimized = t_initial + δt
```

#### Pixel-Based Data Loading

Critical for pose optimization:
- Stores pixel coordinates, not precomputed rays
- Regenerates rays from updated poses each iteration
- Enables proper gradient flow to camera parameters

#### Gradient Flow

```
Pixels → Current Poses → Rays → NeRF → RGB → Loss
           ↑                      ↑
           |                      |
      Pose Optimizer         NeRF Optimizer
```

---

### 📈 Monitoring Training

Training logs include:
- **NeRF metrics**: Loss, PSNR, SSIM, LPIPS
- **Pose errors**: Rotation error (degrees), Translation error (scene units)
- **Convergence**: Per-iteration and per-validation statistics

View in TensorBoard:

```bash
tensorboard --logdir outputs/
```

---

### 🔬 Advanced Usage

See **[POSE_OPTIMIZATION.md](POSE_OPTIMIZATION.md)** for:
- Detailed API documentation
- Configuration options
- Training strategies
- Troubleshooting tips
- Best practices

Example scripts in `scripts/train_pose_optimization.py` demonstrate:
- Clean vs noisy initialization
- Rotation-only optimization
- Translation-only optimization
- Staged optimization strategies
- Extreme noise scenarios


