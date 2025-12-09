"""
Configuration management for NeRF training.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the NeRF model architecture."""
    
    # Positional encoding frequencies
    pos_freqs: int = 10  # L for position encoding (gamma)
    dir_freqs: int = 4   # L for direction encoding
    
    # MLP architecture
    hidden_dim: int = 256
    num_hidden_layers: int = 8
    skips: Tuple[int, ...] = (4,)  # Layers with skip connections
    
    # Output channels
    use_view_dirs: bool = True


@dataclass
class RenderConfig:
    """Configuration for volume rendering."""
    
    # Ray sampling
    near: float = 2.0
    far: float = 6.0
    num_samples: int = 64        # Coarse samples
    num_samples_fine: int = 128  # Fine samples (hierarchical)
    
    # Rendering options
    use_hierarchical: bool = True
    perturb: bool = True        # Stratified sampling jitter during training
    raw_noise_std: float = 0.0  # Noise added to density during training
    
    # White background for synthetic scenes
    white_background: bool = True


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    scene_name: str = "lego"
    data_root: Optional[Path] = None
    img_scale: float = 0.5  # 0.5 = half resolution
    
    # Batching
    batch_size: int = 1024  # Number of rays per batch
    shuffle: bool = True


@dataclass
class TrainConfig:
    """Configuration for training."""
    
    # Optimization
    lr: float = 5e-4
    lr_decay: int = 250  # Exponential decay every N * 1000 iterations
    
    # Training schedule
    num_iterations: int = 200000
    
    # Logging and checkpointing
    log_every: int = 100
    save_every: int = 10000
    val_every: int = 5000
    
    # Output directory
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    experiment_name: str = "baseline"
    
    # Device
    device: str = "cuda"
    
    # Reproducibility
    seed: int = 42


@dataclass
class PoseOptConfig:
    """Configuration for camera pose optimization."""
    
    # Whether to optimize poses
    enabled: bool = True
    
    # What to optimize
    learn_rotation: bool = True
    learn_translation: bool = True
    
    # Optimization settings
    pose_lr: float = 1e-4
    pose_opt_delay: int = 1000  # Start optimizing after N iterations
    
    # Initialization mode
    init_mode: str = "noisy"  # "clean" or "noisy"
    
    # Noise for initialization (if init_mode == "noisy")
    rotation_noise_deg: float = 0.0
    translation_noise_pct: float = 0.0
    noise_seed: Optional[int] = None


@dataclass
class NeRFConfig:
    """Complete configuration for NeRF training."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    pose_opt: Optional[PoseOptConfig] = None  # Optional pose optimization
    
    def __post_init__(self):
        # Convert paths
        if isinstance(self.train.output_dir, str):
            self.train.output_dir = Path(self.train.output_dir)
        if isinstance(self.data.data_root, str):
            self.data.data_root = Path(self.data.data_root)






