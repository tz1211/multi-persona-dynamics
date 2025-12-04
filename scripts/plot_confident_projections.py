#!/usr/bin/env python3
"""Plot projection values for all confident vector variants across checkpoints."""
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_confident_projections(data_dir: str) -> pd.DataFrame:
    """
    Load projection and finetuning shift values for all confident variants from checkpoint CSVs.

    Returns a DataFrame with columns: checkpoint, variant, projection_value, finetuning_shift
    """
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pattern = re.compile(r"checkpoint-(\d+)_confident\.csv$")
    rows = []

    for csv_path in sorted(base.glob("checkpoint-*_confident.csv")):
        match = pattern.match(csv_path.name)
        if not match:
            continue

        checkpoint = int(match.group(1))
        df = pd.read_csv(csv_path)

        # Find all projection columns (confident and confident_X variants)
        # Checkpoint-0 uses a different naming pattern
        if checkpoint == 0:
            proj_pattern = re.compile(
                r"Qwen3-4B-Instruct-2507_(confident(?:_\d+)?)_response_avg_diff_proj_layer20$"
            )
            finetune_pattern = None  # No finetuning shift for checkpoint-0
        else:
            proj_pattern = re.compile(
                rf"checkpoint-{checkpoint}_(confident(?:_\d+)?)_response_avg_diff_proj_layer20$"
            )
            finetune_pattern = re.compile(
                rf"checkpoint-{checkpoint}_(confident(?:_\d+)?)_response_avg_diff_prompt_last_proj_layer20_finetuning_shift$"
            )

        # Collect projection values
        proj_data = {}
        for col in df.columns:
            match = proj_pattern.match(col)
            if match:
                variant = match.group(1)  # e.g., "confident", "confident_2", etc.
                proj_values = pd.to_numeric(df[col], errors="coerce")
                proj_data[variant] = proj_values.mean()

        # Collect finetuning shift values
        finetune_data = {}
        if finetune_pattern is not None:
            for col in df.columns:
                match = finetune_pattern.match(col)
                if match:
                    variant = match.group(1)
                    finetune_values = pd.to_numeric(df[col], errors="coerce")
                    finetune_data[variant] = finetune_values.mean()

        # Combine data for each variant
        for variant in proj_data.keys():
            mean_proj = proj_data[variant]

            # For checkpoint-0, finetuning_shift = 0 (no finetuning yet)
            if checkpoint == 0:
                mean_finetune = 0.0
            else:
                mean_finetune = finetune_data.get(variant, float('nan'))

            rows.append({
                "checkpoint": checkpoint,
                "variant": variant,
                "projection_value": mean_proj,
                "finetuning_shift": mean_finetune
            })

    if not rows:
        raise ValueError(f"No projection data found in {data_dir}")

    return pd.DataFrame(rows).sort_values(["checkpoint", "variant"]).reset_index(drop=True)


def plot_confident_projections(
    data_dir: str,
    save_path: str = None,
    show: bool = False,
    title: str = None
):
    """
    Plot projection values and finetuning shifts for all confident variants across checkpoints.

    Args:
        data_dir: Path to directory containing checkpoint CSVs
        save_path: Optional path to save the plot
        show: Whether to display the plot
        title: Optional custom title for the plot (main title at top of figure)
    """
    # Load data
    df = load_confident_projections(data_dir)

    # Get all variants
    variants = sorted(df["variant"].unique(), key=lambda x: (x != "confident", x))

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Color map for variants
    colors = plt.cm.tab10.colors
    variant_colors = {v: colors[i % len(colors)] for i, v in enumerate(variants)}

    # Plot projections (left subplot)
    for variant in variants:
        variant_df = df[df["variant"] == variant]
        ax1.plot(
            variant_df["checkpoint"],
            variant_df["projection_value"],
            label=variant,
            color=variant_colors[variant],
            marker="o",
            linewidth=2,
            markersize=6
        )

    ax1.set_xlabel("Checkpoint", fontsize=12)
    ax1.set_ylabel("Mean Projection Value", fontsize=12)
    ax1.set_title("Projection Values", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc="best")

    # Plot finetuning shifts (right subplot)
    for variant in variants:
        variant_df = df[df["variant"] == variant]
        ax2.plot(
            variant_df["checkpoint"],
            variant_df["finetuning_shift"],
            label=variant,
            color=variant_colors[variant],
            marker="o",
            linewidth=2,
            markersize=6
        )

    ax2.set_xlabel("Checkpoint", fontsize=12)
    ax2.set_ylabel("Mean Finetuning Shift", fontsize=12)
    ax2.set_title("Finetuning Shift", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc="best")

    # Set main title
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    else:
        fig.suptitle("Confident Vector Analysis Across Checkpoints", fontsize=16, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle

    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()

    return df, (ax1, ax2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot projection values for confident vector variants across checkpoints."
    )
    parser.add_argument(
        "data_dir",
        help="Path to directory containing checkpoint CSVs (e.g., projection_eval/projection_eval/confident_projection_eval)"
    )
    parser.add_argument(
        "--save",
        dest="save_path",
        default=None,
        help="Path to save the plot (e.g., plots/confident_projections.png)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively"
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom title for the plot"
    )

    args = parser.parse_args()

    plot_confident_projections(
        args.data_dir,
        save_path=args.save_path,
        show=args.show,
        title=args.title
    )
