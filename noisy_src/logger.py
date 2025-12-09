"""
Comprehensive logging system for NeRF training.

Supports:
- TensorBoard logging
- CSV logging for easy plotting
- JSON logging for structured data
- Training curve visualization
"""

from __future__ import annotations

import json
import csv
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import torch


@dataclass
class TrainingMetrics:
    """Container for training metrics at a single iteration."""
    iteration: int
    loss: float
    loss_coarse: float
    loss_fine: Optional[float] = None
    psnr: float = 0.0
    learning_rate: float = 0.0
    time_per_iter: float = 0.0
    rays_per_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    iteration: int
    psnr: float
    ssim: float = 0.0
    lpips: Optional[float] = None
    mse: float = 0.0
    
    # Per-image metrics
    per_image_psnr: List[float] = field(default_factory=list)
    per_image_ssim: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Remove None values and empty lists
        return {k: v for k, v in d.items() if v is not None and v != []}


class TensorBoardLogger:
    """TensorBoard logging wrapper."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.writer = None
        self._available = None
    
    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.log_dir)
                self._available = True
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")
                self._available = False
        return self._available
    
    def log_scalar(self, tag: str, value: float, step: int):
        if self.available:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        if self.available:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_image(self, tag: str, img: torch.Tensor, step: int):
        """Log image tensor (H, W, C) in [0, 1]."""
        if self.available:
            # Convert to (C, H, W) for TensorBoard
            if img.dim() == 3:
                img = img.permute(2, 0, 1)
            self.writer.add_image(tag, img, step)
    
    def log_images(self, tag: str, images: List[torch.Tensor], step: int):
        """Log multiple images."""
        if self.available:
            for i, img in enumerate(images):
                self.log_image(f"{tag}/{i}", img, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        if self.available:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        if self.writer is not None:
            self.writer.close()


class CSVLogger:
    """CSV logging for easy pandas/matplotlib analysis."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_file = log_dir / "train_metrics.csv"
        self.val_file = log_dir / "val_metrics.csv"
        
        self._train_writer = None
        self._val_writer = None
        self._train_file_handle = None
        self._val_file_handle = None
    
    def _init_train_csv(self, fieldnames: List[str]):
        self._train_file_handle = open(self.train_file, 'w', newline='')
        self._train_writer = csv.DictWriter(self._train_file_handle, fieldnames=fieldnames)
        self._train_writer.writeheader()
    
    def _init_val_csv(self, fieldnames: List[str]):
        self._val_file_handle = open(self.val_file, 'w', newline='')
        self._val_writer = csv.DictWriter(self._val_file_handle, fieldnames=fieldnames)
        self._val_writer.writeheader()
    
    def log_train(self, metrics: TrainingMetrics):
        data = metrics.to_dict()
        if self._train_writer is None:
            self._init_train_csv(list(data.keys()))
        self._train_writer.writerow(data)
        self._train_file_handle.flush()
    
    def log_val(self, metrics: ValidationMetrics):
        data = metrics.to_dict()
        # Exclude per-image lists for main CSV
        data = {k: v for k, v in data.items() if not isinstance(v, list)}
        if self._val_writer is None:
            self._init_val_csv(list(data.keys()))
        self._val_writer.writerow(data)
        self._val_file_handle.flush()
    
    def close(self):
        if self._train_file_handle:
            self._train_file_handle.close()
        if self._val_file_handle:
            self._val_file_handle.close()


class ExperimentLogger:
    """
    Main experiment logger combining all logging backends.
    
    Usage:
        logger = ExperimentLogger(output_dir)
        
        # During training
        logger.log_training(TrainingMetrics(...))
        
        # During validation
        logger.log_validation(ValidationMetrics(...))
        logger.log_images("val", pred_images, gt_images, iteration)
        
        # End of training
        logger.save_summary()
        logger.close()
    """
    
    def __init__(
        self,
        output_dir: Path,
        experiment_name: str = "experiment",
        use_tensorboard: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.logs_dir = self.output_dir / "logs"
        self.images_dir = self.output_dir / "images"
        self.logs_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        
        # Initialize loggers
        self.csv_logger = CSVLogger(self.logs_dir)
        self.tb_logger = TensorBoardLogger(self.logs_dir / "tensorboard") if use_tensorboard else None
        
        # Track metrics history
        self.train_history: List[TrainingMetrics] = []
        self.val_history: List[ValidationMetrics] = []
        
        # Timing
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Metadata
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "output_dir": str(output_dir),
        }
    
    def log_training(self, metrics: TrainingMetrics):
        """Log training metrics."""
        self.train_history.append(metrics)
        self.csv_logger.log_train(metrics)
        
        if self.tb_logger and self.tb_logger.available:
            self.tb_logger.log_scalar("train/loss", metrics.loss, metrics.iteration)
            self.tb_logger.log_scalar("train/loss_coarse", metrics.loss_coarse, metrics.iteration)
            if metrics.loss_fine is not None:
                self.tb_logger.log_scalar("train/loss_fine", metrics.loss_fine, metrics.iteration)
            self.tb_logger.log_scalar("train/psnr", metrics.psnr, metrics.iteration)
            self.tb_logger.log_scalar("train/learning_rate", metrics.learning_rate, metrics.iteration)
            self.tb_logger.log_scalar("train/rays_per_sec", metrics.rays_per_sec, metrics.iteration)
    
    def log_validation(self, metrics: ValidationMetrics):
        """Log validation metrics."""
        self.val_history.append(metrics)
        self.csv_logger.log_val(metrics)
        
        if self.tb_logger and self.tb_logger.available:
            self.tb_logger.log_scalar("val/psnr", metrics.psnr, metrics.iteration)
            self.tb_logger.log_scalar("val/ssim", metrics.ssim, metrics.iteration)
            self.tb_logger.log_scalar("val/mse", metrics.mse, metrics.iteration)
            if metrics.lpips is not None:
                self.tb_logger.log_scalar("val/lpips", metrics.lpips, metrics.iteration)
    
    def log_images(
        self,
        tag: str,
        pred: torch.Tensor,
        gt: torch.Tensor,
        iteration: int,
        depth: Optional[torch.Tensor] = None,
    ):
        """
        Log predicted and ground truth images.
        
        Parameters
        ----------
        tag : str
            Image tag/name.
        pred : torch.Tensor
            Predicted image (H, W, 3).
        gt : torch.Tensor
            Ground truth image (H, W, 3).
        iteration : int
            Current iteration.
        depth : torch.Tensor, optional
            Depth map (H, W).
        """
        # Save to disk
        self._save_image(pred, self.images_dir / f"{tag}_pred_{iteration:07d}.png")
        self._save_image(gt, self.images_dir / f"{tag}_gt_{iteration:07d}.png")
        
        # Create comparison image (side by side)
        comparison = torch.cat([gt, pred], dim=1)
        self._save_image(comparison, self.images_dir / f"{tag}_comparison_{iteration:07d}.png")
        
        if depth is not None:
            depth_vis = self._depth_to_colormap(depth)
            self._save_image(depth_vis, self.images_dir / f"{tag}_depth_{iteration:07d}.png")
        
        # Log to TensorBoard
        if self.tb_logger and self.tb_logger.available:
            self.tb_logger.log_image(f"{tag}/predicted", pred, iteration)
            self.tb_logger.log_image(f"{tag}/ground_truth", gt, iteration)
            self.tb_logger.log_image(f"{tag}/comparison", comparison, iteration)
            if depth is not None:
                self.tb_logger.log_image(f"{tag}/depth", depth_vis, iteration)
    
    def _save_image(self, img: torch.Tensor, path: Path):
        """Save tensor image to disk."""
        from PIL import Image
        img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np).save(path)
    
    def _depth_to_colormap(self, depth: torch.Tensor) -> torch.Tensor:
        """Convert depth map to colormap visualization."""
        depth = depth.cpu()
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
        
        # Turbo-like colormap
        r = torch.clamp(4 * depth_norm - 1.5, 0, 1)
        g = torch.clamp(2 - 4 * torch.abs(depth_norm - 0.5), 0, 1)
        b = torch.clamp(1.5 - 4 * depth_norm, 0, 1)
        
        return torch.stack([r, g, b], dim=-1)
    
    def log_model_info(self, model: torch.nn.Module, name: str = "model"):
        """Log model architecture and parameter count."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.metadata[f"{name}_total_params"] = total_params
        self.metadata[f"{name}_trainable_params"] = trainable_params
        
        print(f"{name}: {total_params:,} total params, {trainable_params:,} trainable")
    
    def log_config(self, config: Any):
        """Log experiment configuration."""
        if hasattr(config, '__dict__'):
            config_dict = self._config_to_dict(config)
        else:
            config_dict = config
        
        self.metadata["config"] = config_dict
        
        # Save config to file
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def _config_to_dict(self, obj: Any) -> Dict:
        """Recursively convert dataclass config to dict."""
        if hasattr(obj, '__dataclass_fields__'):
            return {k: self._config_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._config_to_dict(v) for v in obj]
        else:
            return obj
    
    def save_summary(self):
        """Save experiment summary including all metrics."""
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["total_time_seconds"] = time.time() - self.start_time
        self.metadata["total_iterations"] = len(self.train_history)
        
        # Final metrics
        if self.val_history:
            final_val = self.val_history[-1]
            self.metadata["final_val_psnr"] = final_val.psnr
            self.metadata["final_val_ssim"] = final_val.ssim
            if final_val.lpips:
                self.metadata["final_val_lpips"] = final_val.lpips
            
            # Best metrics
            best_psnr = max(v.psnr for v in self.val_history)
            best_ssim = max(v.ssim for v in self.val_history)
            self.metadata["best_val_psnr"] = best_psnr
            self.metadata["best_val_ssim"] = best_ssim
        
        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"\nExperiment summary saved to {summary_path}")
    
    def close(self):
        """Close all loggers."""
        self.csv_logger.close()
        if self.tb_logger:
            self.tb_logger.close()


def create_comparison_plot(
    experiments: List[Path],
    metric: str = "psnr",
    output_path: Optional[Path] = None,
):
    """
    Create comparison plot from multiple experiments.
    
    Parameters
    ----------
    experiments : List[Path]
        Paths to experiment output directories.
    metric : str
        Metric to plot ('psnr', 'ssim', 'loss').
    output_path : Path, optional
        Where to save the plot.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib and pandas required for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for exp_path in experiments:
        exp_path = Path(exp_path)
        csv_file = exp_path / "logs" / "train_metrics.csv" if metric == "loss" else exp_path / "logs" / "val_metrics.csv"
        
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found")
            continue
        
        df = pd.read_csv(csv_file)
        label = exp_path.name
        
        if metric in df.columns:
            ax.plot(df["iteration"], df[metric], label=label)
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Training Comparison: {metric.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()







