import os
import glob
import re
from typing import List, Dict, Any

import pandas as pd


def summarize_projection_results(
    trait: str,
    base_output_dir: str = "output",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Summarize trait scores, coherence, projection, and finetuning shift
    across all checkpoints for a given trait.

    This looks for CSV files in:
        output/{trait}_projection_eval/

    Each CSV is expected to have columns like:
        - trait expression score column (e.g., 'critical', 'optimistic', ...)
        - 'coherence'
        - projection columns: '*_proj_layer*'
        - finetuning shift columns: '*finetuning_shift*'

    For each checkpoint file (e.g., 'checkpoint-50_critical.csv'), we compute:
        - mean and std for all trait / coherence columns
        - mean and std for all projection columns
        - mean and std for all finetuning shift columns

    Returns a DataFrame with one row per (checkpoint, judge_trait) pair.
    """
    trait_dir = os.path.join(base_output_dir, f"{trait}_projection_eval")
    if not os.path.isdir(trait_dir):
        raise FileNotFoundError(
            f"Directory '{trait_dir}' not found. "
            f"Expected path: output/{{trait}}_projection_eval/"
        )

    csv_files = sorted(glob.glob(os.path.join(trait_dir, "checkpoint-*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No checkpoint CSVs found in '{trait_dir}'.")

    summary_rows: List[Dict[str, Any]] = []

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        name_no_ext = os.path.splitext(filename)[0]  # e.g., 'checkpoint-50_optimistic'

        # Parse checkpoint number and judge trait from filename
        # Pattern: checkpoint-{step}_{judge_trait}.csv
        m = re.match(r"checkpoint-(\d+)_(.+)", name_no_ext)
        if not m:
            print(f"Warning: Filename '{filename}' does not match expected pattern, skipping.")
            continue

        checkpoint = int(m.group(1))
        judge_trait = m.group(2)

        df = pd.read_csv(csv_path)

        # Identify numeric columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # Trait / coherence score columns: numeric but not checkpoint/projection columns
        score_cols = [
            c for c in numeric_cols
            if not c.startswith("checkpoint-")  # exclude projection / shift columns by prefix
        ]

        # Projection and finetuning shift columns
        proj_cols = [
            c for c in numeric_cols
            if "_proj_layer" in c and "finetuning_shift" not in c
        ]
        shift_cols = [
            c for c in numeric_cols
            if "finetuning_shift" in c
        ]

        row: Dict[str, Any] = {
            "trait_dir": os.path.basename(trait_dir),
            "trait": trait,
            "checkpoint": checkpoint,
            "judge_trait": judge_trait,
        }

        # Trait expression and coherence scores
        for col in score_cols:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std()

        # Projection scores
        for col in proj_cols:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std()

        # Finetuning shift scores
        for col in shift_cols:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    # Sort for easier reading: by checkpoint then judge_trait
    summary_df = summary_df.sort_values(["checkpoint", "judge_trait"]).reset_index(drop=True)

    # Save summary next to the projection eval files
    output_path = os.path.join(trait_dir, "projection_summary.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"Saved projection summary to {output_path}")

    if verbose:
        # Print a human-readable summary to the terminal
        print("\n========== Projection Summary ==========")
        print(f"Trait directory : {os.path.basename(trait_dir)}")
        print(f"Base trait      : {trait}")
        print(f"Checkpoints     : {sorted(summary_df['checkpoint'].unique().tolist())}")
        print(f"Judge traits    : {sorted(summary_df['judge_trait'].unique().tolist())}")
        print("========================================\n")

        # For each checkpoint and judge_trait, show key metrics
        for cp in sorted(summary_df["checkpoint"].unique()):
            cp_df = summary_df[summary_df["checkpoint"] == cp]
            print(f"--- Checkpoint {cp} ---")
            for jt in sorted(cp_df["judge_trait"].unique()):
                row = cp_df[cp_df["judge_trait"] == jt].iloc[0]
                print(f"  Judge trait: {jt}")

                # Trait expression & coherence (if present)
                for col in [jt, "coherence"]:
                    mean_col = f"{col}_mean"
                    std_col = f"{col}_std"
                    if mean_col in row.index and std_col in row.index:
                        print(f"    {col:12s}: mean = {row[mean_col]:8.2f}, std = {row[std_col]:8.2f}")

                # Projection metrics (summarize by average over all projection columns, ignoring NaNs)
                proj_cols_mean = [c for c in summary_df.columns if c.endswith("_mean") and "_proj_layer" in c and "finetuning_shift" not in c]
                if proj_cols_mean:
                    proj_vals = [
                        row[c]
                        for c in proj_cols_mean
                        if c in row.index and pd.notna(row[c])
                    ]
                    if proj_vals:
                        print(f"    projection : avg mean over {len(proj_vals)} vecs = {sum(proj_vals)/len(proj_vals):8.3f}")

                # Finetuning shift metrics (summarize similarly, ignoring NaNs)
                shift_cols_mean = [c for c in summary_df.columns if c.endswith("_mean") and "finetuning_shift" in c]
                if shift_cols_mean:
                    shift_vals = [
                        row[c]
                        for c in shift_cols_mean
                        if c in row.index and pd.notna(row[c])
                    ]
                    if shift_vals:
                        print(f"    shift      : avg mean over {len(shift_vals)} vecs = {sum(shift_vals)/len(shift_vals):8.3f}")

                print()
            print()

    return summary_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize projection results across checkpoints for a given trait."
    )
    parser.add_argument(
        "--trait",
        type=str,
        required=True,
        help="Base trait name, e.g. 'critical' (expects directory output/{trait}_projection_eval/).",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="output",
        help="Base output directory (default: 'output').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a summary table to the terminal.",
    )

    args = parser.parse_args()
    summarize_projection_results(args.trait, base_output_dir=args.base_output_dir, verbose=args.verbose)


