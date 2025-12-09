"""
Example scripts for training NeRF with joint camera pose optimization.

This script demonstrates how to train with different configurations:
1. Clean initialization - verify system maintains performance
2. Noisy initialization - show robustness and pose refinement
3. Rotation-only optimization
4. Translation-only optimization
5. Joint optimization with varying noise levels
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from noisy_src.config import (
    NeRFConfig, ModelConfig, RenderConfig, 
    DataConfig, TrainConfig, PoseOptConfig
)
from noisy_src.train_pose_opt import train_with_pose_optimization
from noisy_src.noise import NoiseConfig


def example_1_clean_initialization():
    """
    Example 1: Train with CLEAN initialization.
    
    This verifies that the pose optimization system maintains performance
    when initialized with ground truth poses (they should stay at GT).
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Clean Initialization (Baseline Verification)")
    print("="*80)
    
    config = NeRFConfig(
        model=ModelConfig(),
        render=RenderConfig(
            use_hierarchical=True,
            num_samples=64,
            num_samples_fine=128,
        ),
        data=DataConfig(
            scene_name="lego",
            img_scale=0.5,
            batch_size=1024,
        ),
        train=TrainConfig(
            lr=5e-4,
            num_iterations=30000,
            log_every=100,
            val_every=2500,
            save_every=10000,
            device="cuda",
        ),
    )
    
    train_with_pose_optimization(
        config=config,
        noise_config=None,
        init_mode="clean",
        pose_lr=1e-4,
        pose_opt_delay=1000,
        learn_rotation=True,
        learn_translation=True,
    )


def example_2_noisy_rotation():
    """
    Example 2: Train with NOISY rotation initialization.
    
    Demonstrates pose refinement with rotation noise only.
    Common scenario: imperfect camera calibration.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Noisy Rotation (2° std)")
    print("="*80)
    
    config = NeRFConfig(
        model=ModelConfig(),
        render=RenderConfig(
            use_hierarchical=True,
            num_samples=64,
            num_samples_fine=128,
        ),
        data=DataConfig(
            scene_name="lego",
            img_scale=0.5,
            batch_size=1024,
        ),
        train=TrainConfig(
            lr=5e-4,
            num_iterations=30000,
            log_every=100,
            val_every=2500,
            save_every=10000,
            device="cuda",
        ),
    )
    
    noise_config = NoiseConfig(
        rotation_noise_deg=2.0,
        translation_noise_pct=0.0,
        seed=42,
    )
    
    train_with_pose_optimization(
        config=config,
        noise_config=noise_config,
        init_mode="noisy",
        pose_lr=1e-4,
        pose_opt_delay=1000,
        learn_rotation=True,
        learn_translation=False,  # Only optimize rotation
    )


def example_3_noisy_translation():
    """
    Example 3: Train with NOISY translation initialization.
    
    Demonstrates pose refinement with translation noise only.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Noisy Translation (1% of camera distance)")
    print("="*80)
    
    config = NeRFConfig(
        model=ModelConfig(),
        render=RenderConfig(
            use_hierarchical=True,
            num_samples=64,
            num_samples_fine=128,
        ),
        data=DataConfig(
            scene_name="lego",
            img_scale=0.5,
            batch_size=1024,
        ),
        train=TrainConfig(
            lr=5e-4,
            num_iterations=30000,
            log_every=100,
            val_every=2500,
            save_every=10000,
            device="cuda",
        ),
    )
    
    noise_config = NoiseConfig(
        rotation_noise_deg=0.0,
        translation_noise_pct=1.0,
        seed=42,
    )
    
    train_with_pose_optimization(
        config=config,
        noise_config=noise_config,
        init_mode="noisy",
        pose_lr=1e-4,
        pose_opt_delay=1000,
        learn_rotation=False,  # Only optimize translation
        learn_translation=True,
    )


def example_4_joint_optimization():
    """
    Example 4: Train with BOTH rotation and translation noise.
    
    Full joint optimization of all camera parameters.
    This is the most realistic scenario.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Joint Optimization (Rotation + Translation)")
    print("="*80)
    
    config = NeRFConfig(
        model=ModelConfig(),
        render=RenderConfig(
            use_hierarchical=True,
            num_samples=64,
            num_samples_fine=128,
        ),
        data=DataConfig(
            scene_name="lego",
            img_scale=0.5,
            batch_size=1024,
        ),
        train=TrainConfig(
            lr=5e-4,
            num_iterations=50000,  # Longer training for joint opt
            log_every=100,
            val_every=2500,
            save_every=10000,
            device="cuda",
        ),
    )
    
    noise_config = NoiseConfig(
        rotation_noise_deg=2.0,
        translation_noise_pct=1.0,
        seed=42,
    )
    
    train_with_pose_optimization(
        config=config,
        noise_config=noise_config,
        init_mode="noisy",
        pose_lr=1e-4,
        pose_opt_delay=1000,
        learn_rotation=True,
        learn_translation=True,
    )


def example_5_severe_noise():
    """
    Example 5: Train with SEVERE noise to test limits.
    
    Tests the system with extreme noise levels.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Severe Noise (5° rotation + 2% translation)")
    print("="*80)
    
    config = NeRFConfig(
        model=ModelConfig(),
        render=RenderConfig(
            use_hierarchical=True,
            num_samples=64,
            num_samples_fine=128,
        ),
        data=DataConfig(
            scene_name="lego",
            img_scale=0.5,
            batch_size=1024,
        ),
        train=TrainConfig(
            lr=5e-4,
            num_iterations=50000,
            log_every=100,
            val_every=2500,
            save_every=10000,
            device="cuda",
        ),
    )
    
    noise_config = NoiseConfig(
        rotation_noise_deg=5.0,
        translation_noise_pct=2.0,
        seed=42,
    )
    
    train_with_pose_optimization(
        config=config,
        noise_config=noise_config,
        init_mode="noisy",
        pose_lr=2e-4,  # Higher LR for severe noise
        pose_opt_delay=500,  # Start earlier
        learn_rotation=True,
        learn_translation=True,
    )


def example_6_delayed_pose_optimization():
    """
    Example 6: Two-stage training.
    
    First train NeRF with frozen poses, then optimize poses.
    This can be more stable in some cases.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Delayed Pose Optimization")
    print("="*80)
    
    config = NeRFConfig(
        model=ModelConfig(),
        render=RenderConfig(
            use_hierarchical=True,
            num_samples=64,
            num_samples_fine=128,
        ),
        data=DataConfig(
            scene_name="lego",
            img_scale=0.5,
            batch_size=1024,
        ),
        train=TrainConfig(
            lr=5e-4,
            num_iterations=40000,
            log_every=100,
            val_every=2500,
            save_every=10000,
            device="cuda",
        ),
    )
    
    noise_config = NoiseConfig(
        rotation_noise_deg=2.0,
        translation_noise_pct=1.0,
        seed=42,
    )
    
    train_with_pose_optimization(
        config=config,
        noise_config=noise_config,
        init_mode="noisy",
        pose_lr=1e-4,
        pose_opt_delay=10000,  # Wait 10k iterations before pose opt
        learn_rotation=True,
        learn_translation=True,
    )


def main():
    """Run selected examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pose optimization examples")
    parser.add_argument(
        "--example",
        type=int,
        default=4,
        choices=[1, 2, 3, 4, 5, 6],
        help="Which example to run (1-6)"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        help="Scene name to use"
    )
    
    args = parser.parse_args()
    
    # Override scene in all examples
    # (In practice, you'd modify the functions to accept scene as parameter)
    
    examples = {
        1: example_1_clean_initialization,
        2: example_2_noisy_rotation,
        3: example_3_noisy_translation,
        4: example_4_joint_optimization,
        5: example_5_severe_noise,
        6: example_6_delayed_pose_optimization,
    }
    
    print(f"\nRunning Example {args.example}...")
    examples[args.example]()


if __name__ == "__main__":
    main()
