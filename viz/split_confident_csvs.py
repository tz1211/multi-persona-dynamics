#!/usr/bin/env python3
"""Split confident CSV files into separate files for each variant."""
import pandas as pd
from pathlib import Path
import sys

def split_confident_csvs(input_dir: str, output_base_dir: str):
    """
    Split confident CSV files containing multiple variants into separate files.

    Args:
        input_dir: Directory containing checkpoint-*_confident.csv files
        output_base_dir: Base directory for output (will create subdirectories)
    """
    input_path = Path(input_dir)

    # Confident variants to extract
    variants = ["confident", "confident_2", "confident_3", "confident_4", "confident_5"]

    # Create output directories for each variant
    output_dirs = {}
    for variant in variants:
        variant_dir = Path(output_base_dir) / f"{variant}_projection_eval"
        variant_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[variant] = variant_dir
        print(f"Created directory: {variant_dir}")

    # Process each CSV file
    csv_files = sorted(input_path.glob("checkpoint-*_confident.csv"))
    print(f"\nProcessing {len(csv_files)} CSV files...")

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        checkpoint_num = csv_file.name.split("_")[0]  # e.g., "checkpoint-50"

        print(f"\nProcessing {csv_file.name}...")

        for variant in variants:
            # Select columns for this variant
            base_cols = ["question", "prompt", "answer", "question_id"]

            # Persona score column
            if variant == "confident":
                persona_col = "confident"
            else:
                persona_col = variant  # confident_2, confident_3, etc.

            # Coherence column (same for all)
            coherence_col = "coherence"

            # Projection and finetuning shift columns
            proj_col = f"{checkpoint_num}_{variant}_response_avg_diff_proj_layer20"
            finetune_col = f"{checkpoint_num}_{variant}_response_avg_diff_prompt_last_proj_layer20_finetuning_shift"

            # Check if columns exist
            required_cols = [persona_col, coherence_col, proj_col, finetune_col]
            missing = [col for col in required_cols if col not in df.columns]

            if missing:
                print(f"  ⚠️  {variant}: Missing columns {missing}, skipping...")
                continue

            # Create subset DataFrame
            subset_df = df[base_cols + [persona_col, coherence_col, proj_col, finetune_col]].copy()

            # Rename columns to match expected format (without variant suffix for the main persona column)
            subset_df = subset_df.rename(columns={
                persona_col: variant,
                proj_col: f"{checkpoint_num}_{variant}_response_avg_diff_proj_layer20",
                finetune_col: f"{checkpoint_num}_{variant}_response_avg_diff_prompt_last_proj_layer20_finetuning_shift"
            })

            # Save to output directory
            output_file = output_dirs[variant] / f"{checkpoint_num}_{variant}.csv"
            subset_df.to_csv(output_file, index=False)
            print(f"  ✓ {variant}: Saved {len(subset_df)} rows to {output_file.name}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_confident_csvs.py <input_dir> <output_base_dir>")
        print("Example: python split_confident_csvs.py projection_eval/confident_projection_eval projection_eval")
        sys.exit(1)

    split_confident_csvs(sys.argv[1], sys.argv[2])
    print("\n✅ Done! You can now plot each variant separately.")
