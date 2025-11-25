## Noise-Robust NeRF Camera Calibration

This repository contains the scaffolding for a class final project on **noise-robust Neural Radiance Fields (NeRF) for camera calibration**, using the **NeRF Synthetic Blender dataset (lego scene)**.

The focus of this project is to study how errors in camera intrinsics and extrinsics affect NeRF reconstruction quality, and how to **jointly optimize camera parameters and NeRF** to be robust to such noise.

### Project Goals

- **Baseline NeRF**: Implement and train a standard NeRF model on the lego scene with accurate camera parameters.
- **Fixed-noisy NeRF**: Inject controlled noise into camera intrinsics/extrinsics and train NeRF while keeping noisy parameters fixed.
- **Joint optimization**: Treat camera parameters as learnable variables and jointly optimize them with NeRF using reconstruction loss, starting from noisy initializations.

This repository currently provides **structure and utilities only**. The actual NeRF model and training logic will be implemented later.

---

### Repository Structure

- **`data/`**
  - **`raw/`**: Raw NeRF Blender assets (e.g., images and `transforms*.json`).
  - **`processed/`**: Any preprocessed or cached data (rays, feature tensors, etc.).
- **`scripts/`**
  - **`download_blender_dataset.py`**: Helper script to download and extract the NeRF Synthetic Blender lego scene.
  - **`visualize_dataset.py`**: Script to quickly inspect images and visualize camera poses in 3D.
  - **`inject_noise.py`**: Simple Gaussian-noise utilities for intrinsics and extrinsics (for experimentation and prototyping).
- **`src/`**
  - **`camera/`**
    - `camera_parameters.py`: Placeholder container class for camera intrinsics/extrinsics.
    - `noise_models.py`: Placeholder API for more structured camera noise models.
  - **`nerf/`**
    - `model.py`: NeRF model stub (no implementation yet).
    - `rendering.py`: Placeholder for volume rendering utilities.
  - **`training/`**
    - `train_baseline.py`: Entry point stub for baseline NeRF training.
    - `train_noisy_fixed.py`: Entry point stub for fixed-noisy camera experiments.
    - `train_joint_optimization.py`: Entry point stub for joint NeRF + camera optimization.
- **`notebooks/`**
  - **`explore_data.ipynb`**: Explore images and camera poses for the lego scene.
  - **`visualize_noise_effects.ipynb`**: Prototype and visualize the effect of noisy camera parameters.
- **`requirements.txt`**: Python dependencies for utilities, notebooks, and (later) NeRF implementation.

---

### Dataset: NeRF Synthetic Blender (lego)

This project uses the **NeRF Synthetic Blender dataset**, specifically the **lego** scene.

For each scene, the dataset typically provides:

- **RGB images** (often split into `train/`, `val/`, `test/`).
- **Camera transforms**:
  - A JSON file (e.g., `transforms.json` or `transforms_train.json`) containing **camera-to-world** transform matrices.
  - Approximate **field-of-view (FOV)** information.

The helper script in `scripts/download_blender_dataset.py` is set up to download a **ZIP archive** of the lego scene into `data/raw/lego/`, but you will need to set a working **direct download URL** for the dataset.

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

#### 2. Download the lego dataset

Update the `DATASET_URL` constant inside `scripts/download_blender_dataset.py` to point to a **direct ZIP download** of the NeRF Blender lego scene (containing images and `transforms*.json`).

Then run:

```bash
python scripts/download_blender_dataset.py
```

By default this will extract the dataset under:

- **`data/raw/lego/`**

If you already have the dataset downloaded manually, you can simply place it under `data/raw/lego/` with:

- images under `data/raw/lego/` or `data/raw/lego/train/`
- JSON transforms file (e.g., `transforms.json` or `transforms_train.json`) in `data/raw/lego/`

---

### Running the Visualization Script

To quickly sanity-check that the dataset is in the expected format, you can run:

```bash
python scripts/visualize_dataset.py
```

This will:

- Display a few sample images from the lego scene.
- Plot camera centers in 3D using the transform matrices from `transforms.json` or `transforms_train.json`.

---

### Using the Notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then open:

- **`notebooks/explore_data.ipynb`**
  - Inspect sample RGB frames.
  - Visualize the distribution of camera poses.
- **`notebooks/visualize_noise_effects.ipynb`**
  - Load camera transforms.
  - Apply simple Gaussian noise to extrinsics.
  - Visualize original vs noisy camera trajectories.

---

### Planned Experiments

- **Baseline NeRF**
  - Implement a standard NeRF MLP with positional encoding.
  - Train on the lego scene using ground-truth camera parameters.
  - Evaluate reconstruction quality (PSNR, qualitative views).

- **Fixed-noisy NeRF**
  - Inject synthetic Gaussian noise into:
    - **Intrinsics** (e.g., focal length, principal point).
    - **Extrinsics** (camera-to-world poses).
  - Train NeRF with these **fixed noisy parameters**.
  - Compare reconstruction metrics vs the baseline.

- **Joint optimization**
  - Introduce **learnable** camera parameters (e.g., per-view pose, intrinsics).
  - Jointly optimize NeRF and camera parameters using reconstruction loss.
  - Study convergence behavior and robustness to different noise levels.

---

### TODO: NeRF and Training Implementation

- **NeRF model**
  - Implement positional encoding for 3D positions and view directions.
  - Implement the NeRF MLP (coarse + fine networks, or a single network baseline).
  - Add volume rendering utilities in `src/nerf/rendering.py`.

- **Training pipeline**
  - Implement data loading and ray sampling utilities.
  - Implement baseline training in `src/training/train_baseline.py`.
  - Implement fixed-noisy experiments in `src/training/train_noisy_fixed.py`.
  - Implement joint optimization in `src/training/train_joint_optimization.py`.

- **Camera + noise**
  - Flesh out `src/camera/camera_parameters.py` for camera parameter management.
  - Implement reusable noise models in `src/camera/noise_models.py`.
  - Integrate noise injection and camera parameter optimization into training loops.


