"""
Visualization utilities for NeRF experiments.

Creates publication-ready plots for:
- Training curves (loss, PSNR)
- Validation metrics over time
- Multi-experiment comparisons
- Per-image analysis
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Check for pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def check_dependencies():
    """Check if visualization dependencies are available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required: pip install matplotlib")
    if not HAS_PANDAS:
        raise ImportError("pandas required: pip install pandas")


def load_training_logs(exp_dir: Path) -> pd.DataFrame:
    """Load training metrics from CSV."""
    check_dependencies()
    csv_path = exp_dir / "logs" / "train_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training log not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_validation_logs(exp_dir: Path) -> pd.DataFrame:
    """Load validation metrics from CSV."""
    check_dependencies()
    csv_path = exp_dir / "logs" / "val_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation log not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_summary(exp_dir: Path) -> Dict[str, Any]:
    """Load experiment summary."""
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    with open(summary_path) as f:
        return json.load(f)


def plot_training_curves(
    exp_dir: Path,
    output_path: Optional[Path] = None,
    show: bool = True,
    figsize: tuple = (14, 10),
):
    """
    Plot comprehensive training curves for a single experiment.
    
    Creates a 2x2 grid with:
    - Training loss (coarse + fine)
    - Training PSNR
    - Learning rate schedule
    - Training speed (rays/sec)
    """
    check_dependencies()
    
    df = load_training_logs(Path(exp_dir))
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Training Progress: {Path(exp_dir).name}", fontsize=14, fontweight='bold')
    
    # Smooth function for cleaner plots
    def smooth(data, window=100):
        return pd.Series(data).rolling(window=window, min_periods=1).mean()
    
    # 1. Training Loss
    ax = axes[0, 0]
    ax.plot(df['iteration'], smooth(df['loss']), label='Total Loss', color='blue', alpha=0.8)
    ax.plot(df['iteration'], smooth(df['loss_coarse']), label='Coarse Loss', color='orange', alpha=0.7)
    if 'loss_fine' in df.columns and df['loss_fine'].notna().any():
        ax.plot(df['iteration'], smooth(df['loss_fine'].fillna(0)), label='Fine Loss', color='green', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 2. Training PSNR
    ax = axes[0, 1]
    ax.plot(df['iteration'], smooth(df['psnr']), color='blue', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Training PSNR')
    ax.grid(True, alpha=0.3)
    
    # 3. Learning Rate
    ax = axes[1, 0]
    ax.plot(df['iteration'], df['learning_rate'], color='purple')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 4. Training Speed
    ax = axes[1, 1]
    ax.plot(df['iteration'], smooth(df['rays_per_sec']), color='teal')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Rays/second')
    ax.set_title('Training Speed')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_validation_curves(
    exp_dir: Path,
    output_path: Optional[Path] = None,
    show: bool = True,
    figsize: tuple = (12, 4),
):
    """
    Plot validation metrics over training.
    
    Shows PSNR, SSIM, and optionally LPIPS.
    """
    check_dependencies()
    
    df = load_validation_logs(Path(exp_dir))
    
    has_lpips = 'lpips' in df.columns and df['lpips'].notna().any()
    ncols = 3 if has_lpips else 2
    
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    fig.suptitle(f"Validation Metrics: {Path(exp_dir).name}", fontsize=14, fontweight='bold')
    
    # PSNR
    ax = axes[0]
    ax.plot(df['iteration'], df['psnr'], 'o-', color='blue', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Validation PSNR')
    ax.grid(True, alpha=0.3)
    
    # Add best value annotation
    best_idx = df['psnr'].idxmax()
    best_iter = df.loc[best_idx, 'iteration']
    best_psnr = df.loc[best_idx, 'psnr']
    ax.axhline(y=best_psnr, color='red', linestyle='--', alpha=0.5)
    ax.annotate(f'Best: {best_psnr:.2f} dB', 
                xy=(best_iter, best_psnr), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color='red')
    
    # SSIM
    ax = axes[1]
    ax.plot(df['iteration'], df['ssim'], 'o-', color='green', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('SSIM')
    ax.set_title('Validation SSIM')
    ax.grid(True, alpha=0.3)
    
    # LPIPS (if available)
    if has_lpips:
        ax = axes[2]
        ax.plot(df['iteration'], df['lpips'], 'o-', color='orange', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('LPIPS')
        ax.set_title('Validation LPIPS (lower is better)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved validation curves to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compare_experiments(
    exp_dirs: List[Path],
    metric: str = "psnr",
    val_or_train: str = "val",
    labels: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Compare multiple experiments on a single plot.
    
    Parameters
    ----------
    exp_dirs : List[Path]
        List of experiment directories.
    metric : str
        Metric to compare ('psnr', 'ssim', 'lpips', 'loss').
    val_or_train : str
        Use validation ('val') or training ('train') metrics.
    labels : List[str], optional
        Custom labels for each experiment.
    output_path : Path, optional
        Where to save the plot.
    show : bool
        Whether to display the plot.
    """
    check_dependencies()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_dirs)))
    
    for i, exp_dir in enumerate(exp_dirs):
        exp_dir = Path(exp_dir)
        label = labels[i] if labels else exp_dir.name
        
        try:
            if val_or_train == "val":
                df = load_validation_logs(exp_dir)
                linestyle = 'o-'
            else:
                df = load_training_logs(exp_dir)
                linestyle = '-'
                # Smooth training data
                df[metric] = pd.Series(df[metric]).rolling(window=100, min_periods=1).mean()
            
            if metric in df.columns:
                ax.plot(df['iteration'], df[metric], linestyle, 
                       label=label, color=colors[i], markersize=3, alpha=0.8)
            else:
                print(f"Warning: Metric '{metric}' not found in {exp_dir}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Experiment Comparison: {metric.upper()} ({val_or_train})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_summary_table(
    exp_dirs: List[Path],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create a summary table comparing final metrics across experiments.
    
    Returns a DataFrame and optionally saves to CSV.
    """
    check_dependencies()
    
    rows = []
    for exp_dir in exp_dirs:
        exp_dir = Path(exp_dir)
        try:
            summary = load_summary(exp_dir)
            row = {
                'Experiment': exp_dir.name,
                'Final PSNR': summary.get('final_val_psnr', '-'),
                'Best PSNR': summary.get('best_val_psnr', '-'),
                'Final SSIM': summary.get('final_val_ssim', '-'),
                'Best SSIM': summary.get('best_val_ssim', '-'),
                'Final LPIPS': summary.get('final_val_lpips', '-'),
                'Total Time (h)': summary.get('total_time_seconds', 0) / 3600,
                'Iterations': summary.get('total_iterations', '-'),
            }
            rows.append(row)
        except FileNotFoundError:
            print(f"Warning: Summary not found for {exp_dir}")
    
    df = pd.DataFrame(rows)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved summary table to {output_path}")
    
    return df


def plot_final_results(
    exp_dirs: List[Path],
    output_path: Optional[Path] = None,
    show: bool = True,
    figsize: tuple = (12, 5),
):
    """
    Create bar chart comparing final metrics across experiments.
    """
    check_dependencies()
    
    df = create_summary_table(exp_dirs)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Final Results Comparison', fontsize=14, fontweight='bold')
    
    x = np.arange(len(df))
    width = 0.6
    
    # PSNR
    ax = axes[0]
    psnr_vals = pd.to_numeric(df['Best PSNR'], errors='coerce')
    bars = ax.bar(x, psnr_vals, width, color='steelblue')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Best Validation PSNR')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Experiment'], rotation=45, ha='right')
    ax.bar_label(bars, fmt='%.2f', fontsize=8)
    
    # SSIM
    ax = axes[1]
    ssim_vals = pd.to_numeric(df['Best SSIM'], errors='coerce')
    bars = ax.bar(x, ssim_vals, width, color='seagreen')
    ax.set_ylabel('SSIM')
    ax.set_title('Best Validation SSIM')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Experiment'], rotation=45, ha='right')
    ax.bar_label(bars, fmt='%.4f', fontsize=8)
    
    # Training Time
    ax = axes[2]
    time_vals = pd.to_numeric(df['Total Time (h)'], errors='coerce')
    bars = ax.bar(x, time_vals, width, color='coral')
    ax.set_ylabel('Time (hours)')
    ax.set_title('Training Time')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Experiment'], rotation=45, ha='right')
    ax.bar_label(bars, fmt='%.2f', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved final results plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_report_figures(
    exp_dir: Path,
    output_dir: Optional[Path] = None,
):
    """
    Generate all figures needed for a report from a single experiment.
    
    Creates:
    - training_curves.png
    - validation_curves.png
    """
    check_dependencies()
    
    exp_dir = Path(exp_dir)
    if output_dir is None:
        output_dir = exp_dir / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating report figures for: {exp_dir.name}")
    print(f"Output directory: {output_dir}")
    
    # Training curves
    try:
        plot_training_curves(
            exp_dir,
            output_path=output_dir / "training_curves.png",
            show=False
        )
    except Exception as e:
        print(f"  Warning: Could not generate training curves: {e}")
    
    # Validation curves
    try:
        plot_validation_curves(
            exp_dir,
            output_path=output_dir / "validation_curves.png",
            show=False
        )
    except Exception as e:
        print(f"  Warning: Could not generate validation curves: {e}")
    
    print("Done!")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NeRF Visualization Tools")
    subparsers = parser.add_subparsers(dest="command")
    
    # Training curves
    train_parser = subparsers.add_parser("training", help="Plot training curves")
    train_parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    train_parser.add_argument("--output", type=Path, help="Output path")
    train_parser.add_argument("--no-show", action="store_true")
    
    # Validation curves  
    val_parser = subparsers.add_parser("validation", help="Plot validation curves")
    val_parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    val_parser.add_argument("--output", type=Path, help="Output path")
    val_parser.add_argument("--no-show", action="store_true")
    
    # Compare experiments
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("exp_dirs", type=Path, nargs="+", help="Experiment directories")
    compare_parser.add_argument("--metric", default="psnr", help="Metric to compare")
    compare_parser.add_argument("--output", type=Path, help="Output path")
    compare_parser.add_argument("--no-show", action="store_true")
    
    # Report figures
    report_parser = subparsers.add_parser("report", help="Generate all report figures")
    report_parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    report_parser.add_argument("--output-dir", type=Path, help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == "training":
        plot_training_curves(args.exp_dir, args.output, show=not args.no_show)
    elif args.command == "validation":
        plot_validation_curves(args.exp_dir, args.output, show=not args.no_show)
    elif args.command == "compare":
        compare_experiments(args.exp_dirs, args.metric, output_path=args.output, show=not args.no_show)
    elif args.command == "report":
        create_report_figures(args.exp_dir, args.output_dir)
    else:
        parser.print_help()







