#!/usr/bin/env python3
"""
Analyze and visualize evaluation results across checkpoints
Usage: python scripts/analyze_checkpoint_results.py --results_dir results/anxious_checkpoints
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import json


def extract_checkpoint_number(filename):
    """Extract checkpoint number from filename like 'checkpoint-50.csv'"""
    match = re.search(r'checkpoint-(\d+)', filename)
    return int(match.group(1)) if match else None


def load_checkpoint_results(results_dir):
    """Load all checkpoint results from a directory"""
    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise ValueError(f"Results directory not found: {results_dir}")

    all_results = []

    for csv_file in sorted(results_dir.glob("checkpoint-*.csv")):
        checkpoint_num = extract_checkpoint_number(csv_file.name)
        if checkpoint_num is None:
            continue

        try:
            df = pd.read_csv(csv_file)

            # Calculate summary statistics
            summary = {
                'checkpoint': checkpoint_num,
                'checkpoint_name': csv_file.stem,
                'file_path': str(csv_file)
            }

            # Add mean and std for each metric column
            for col in df.columns:
                if col not in ['question', 'prompt', 'answer', 'question_id']:
                    summary[f'{col}_mean'] = df[col].mean()
                    summary[f'{col}_std'] = df[col].std()
                    summary[f'{col}_min'] = df[col].min()
                    summary[f'{col}_max'] = df[col].max()

            all_results.append(summary)

        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
            continue

    if not all_results:
        raise ValueError(f"No valid checkpoint results found in {results_dir}")

    # Sort by checkpoint number
    all_results.sort(key=lambda x: x['checkpoint'])

    return pd.DataFrame(all_results)


def print_summary(summary_df, trait_name=None):
    """Print a formatted summary of results"""
    print("\n" + "="*80)
    print("CHECKPOINT EVALUATION SUMMARY")
    print("="*80)
    print(f"\nTotal checkpoints analyzed: {len(summary_df)}")
    print(f"Checkpoint range: {summary_df['checkpoint'].min()} - {summary_df['checkpoint'].max()}")

    # Identify metrics (columns ending with _mean)
    metric_cols = [col for col in summary_df.columns if col.endswith('_mean')]
    metrics = [col.replace('_mean', '') for col in metric_cols]

    print(f"\nMetrics found: {', '.join(metrics)}")
    print("\n" + "-"*80)

    # Print detailed statistics for each metric
    for metric in metrics:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'

        if mean_col in summary_df.columns:
            print(f"\n{metric.upper()} Scores:")
            print("-" * 40)

            for _, row in summary_df.iterrows():
                checkpoint_str = f"Checkpoint-{row['checkpoint']:>4}"
                mean_val = row[mean_col]
                std_val = row.get(std_col, 0)

                # Add visual bar
                bar_length = int(mean_val / 2)  # Assuming 0-100 scale
                bar = '█' * bar_length

                print(f"{checkpoint_str}: {mean_val:6.2f} ± {std_val:5.2f}  {bar}")

            # Print statistics across checkpoints
            print(f"\nAcross all checkpoints:")
            print(f"  Best:  {summary_df[mean_col].max():.2f} (checkpoint-{summary_df.loc[summary_df[mean_col].idxmax(), 'checkpoint']:.0f})")
            print(f"  Worst: {summary_df[mean_col].min():.2f} (checkpoint-{summary_df.loc[summary_df[mean_col].idxmin(), 'checkpoint']:.0f})")
            print(f"  Mean:  {summary_df[mean_col].mean():.2f}")
            print(f"  Std:   {summary_df[mean_col].std():.2f}")

    # Check for projection columns
    proj_cols = [col for col in summary_df.columns if 'proj' in col.lower() and col.endswith('_mean')]
    if proj_cols:
        print("\n" + "="*80)
        print("PROJECTION ANALYSIS")
        print("="*80)
        for proj_col in proj_cols:
            metric_name = proj_col.replace('_mean', '')
            print(f"\n{metric_name}:")
            print("-" * 40)
            for _, row in summary_df.iterrows():
                checkpoint_str = f"Checkpoint-{row['checkpoint']:>4}"
                proj_val = row[proj_col]
                print(f"{checkpoint_str}: {proj_val:8.4f}")

    print("\n" + "="*80)


def save_summary(summary_df, output_path):
    """Save summary to CSV and JSON"""
    # Save as CSV
    csv_path = Path(output_path).with_suffix('.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    # Save as JSON for easy parsing
    json_path = Path(output_path).with_suffix('.json')
    summary_df.to_json(json_path, orient='records', indent=2)
    print(f"Summary saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze checkpoint evaluation results')
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing checkpoint CSV files')
    parser.add_argument('--output', type=str, default=None,
                      help='Output path for summary (without extension, will create .csv and .json)')
    parser.add_argument('--trait', type=str, default=None,
                      help='Trait name being evaluated (for display purposes)')

    args = parser.parse_args()

    # Load and analyze results
    print(f"Loading results from: {args.results_dir}")
    summary_df = load_checkpoint_results(args.results_dir)

    # Print summary
    print_summary(summary_df, args.trait)

    # Save summary if output path provided
    if args.output:
        save_summary(summary_df, args.output)
    else:
        # Default output path
        default_output = Path(args.results_dir) / "summary"
        save_summary(summary_df, default_output)


if __name__ == "__main__":
    main()
