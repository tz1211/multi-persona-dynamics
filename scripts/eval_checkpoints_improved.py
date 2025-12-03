#!/usr/bin/env python3
"""
Improved parallel checkpoint evaluation using EXISTING scripts with minimal modifications.

This script:
1. Uses eval_persona.py with --judge_traits to evaluate with multiple trait judges
2. Uses cal_projection.py to calculate projections for all traits at once
3. Uses cal_projection.py with --base_model to calculate finetuning shifts (per-example)
4. Processes multiple checkpoints in parallel across multiple GPUs

All results are saved in a single CSV per checkpoint with columns for:
- Judge scores for each trait (e.g., confident, critical, anxious, sycophantic)
- Projections for each trait (e.g., checkpoint-100_confident_proj_layer20)
- Finetuning shifts for each trait (e.g., checkpoint-100_confident_prompt_last_proj_layer20_finetuning_shift)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def process_checkpoint(checkpoint_path, gpu_id, args):
    """Process a single checkpoint with multi-trait evaluation."""
    checkpoint_name = os.path.basename(checkpoint_path)
    output_path = os.path.join(args.results_dir, f"{checkpoint_name}.csv")
    log_file = os.path.join(args.results_dir, f"{checkpoint_name}.log")

    log_messages = []

    def log(msg):
        log_messages.append(msg)
        logger.info(f"[{checkpoint_name}] {msg}")

    log(f"Processing on GPU {gpu_id}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # Step 1: Multi-trait LLM judge evaluation using EXISTING eval_persona.py
        log("Step 1/3: Running multi-trait LLM judge evaluation...")
        log(f"  Question trait: {args.question_trait}")
        log(f"  Judge traits: {' '.join(args.judge_traits)}")

        # Join judge traits with commas
        judge_traits_str = ",".join(args.judge_traits)
        cmd = [
            "python", "-m", "eval.eval_persona",
            "--model", checkpoint_path,
            "--trait", args.question_trait,
            "--judge_traits", judge_traits_str,  
            "--output_path", output_path,
            "--judge_model", args.judge_model,
            "--version", args.version,
            "--n_per_question", str(args.n_per_question)
        ]

        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=os.getcwd())

        if result.returncode != 0:
            log(f"✗ Evaluation failed: {result.stderr}")
            return checkpoint_name, False, "\n".join(log_messages)

        log("✓ Multi-trait evaluation completed")

        # Step 2: Calculate projections for ALL traits at once using EXISTING cal_projection.py
        log("Step 2/3: Calculating projections for all traits...")

        vector_paths = []
        layer_list = []
        for trait in args.judge_traits:
            vector_path = f"output/persona_vectors/Qwen/Qwen3-4B-Instruct-2507/{trait}_response_avg_diff.pt"
            if os.path.exists(vector_path):
                vector_paths.append(vector_path)
                layer_list.append(str(args.layer))
            else:
                log(f"⚠ Vector not found for {trait}: {vector_path}")

        if vector_paths:
            cmd = [
                "python", "-m", "eval.cal_projection",
                "--file_path", output_path,
                "--vector_path_list", *vector_paths,  # Pass ALL vectors at once!
                "--layer_list", *layer_list,
                "--model_name", checkpoint_path,
                "--projection_type", args.projection_type
            ]

            result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=os.getcwd())

            if result.returncode != 0:
                log(f"✗ Projection failed: {result.stderr}")
            else:
                log("✓ Projection completed")

        # Step 3: Calculate finetuning shift using cal_projection.py with --base_model
        log("Step 3/3: Calculating finetuning shift (checkpoint - base) for all traits...")

        if vector_paths:
            cmd = [
                "python", "-m", "eval.cal_projection",
                "--file_path", output_path,
                "--vector_path_list", *vector_paths,  # Same vectors as Step 2
                "--layer_list", *layer_list,
                "--model_name", checkpoint_path,
                "--base_model", args.base_model,  # This makes it calculate (finetuned - base)!
                "--projection_type", "prompt_last_proj"  # Use last prompt token for shift
            ]

            result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=os.getcwd())

            if result.returncode != 0:
                log(f"✗ Finetuning shift failed: {result.stderr}")
            else:
                log("✓ Finetuning shift completed (per-example shifts calculated)")

        log("Processing complete!")

        with open(log_file, 'w') as f:
            f.write("\n".join(log_messages))

        return checkpoint_name, True, "\n".join(log_messages)

    except Exception as e:
        log(f"✗ Exception occurred: {str(e)}")
        return checkpoint_name, False, "\n".join(log_messages)


def main():
    parser = argparse.ArgumentParser(
        description="Improved parallel checkpoint evaluation using EXISTING scripts"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing checkpoint-* subdirectories")
    parser.add_argument("--question_trait", type=str, required=True,
                        help="Trait to use for questions (e.g., 'confident')")
    parser.add_argument("--judge_traits", type=str, nargs="+", required=True,
                        help="Traits to judge with (e.g., 'confident critical anxious sycophantic')")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer to extract activations from")
    parser.add_argument("--n_per_question", type=int, default=100,
                        help="Number of samples per question")
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of GPUs to use for parallel processing")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen3-4B-Instruct-2507",
                        help="Base model path")
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini-2025-04-14",
                        help="Judge model")
    parser.add_argument("--version", type=str, default="eval",
                        help="Evaluation data version")
    parser.add_argument("--projection_type", type=str, default="proj",
                        help="Projection type")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results directory (default: results/<checkpoint_dir_name>)")

    args = parser.parse_args()

    if args.results_dir is None:
        checkpoint_dir_name = os.path.basename(args.checkpoint_dir.rstrip('/'))
        args.results_dir = f"results/{checkpoint_dir_name}"

    os.makedirs(args.results_dir, exist_ok=True)

    main_log_file = os.path.join(args.results_dir, f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    logger.info("========================================")
    logger.info("Improved Parallel Multi-Trait Checkpoint Evaluation")
    logger.info("Using EXISTING scripts with minimal modifications!")
    logger.info("========================================")
    logger.info(f"Checkpoint Directory: {args.checkpoint_dir}")
    logger.info(f"Question Trait: {args.question_trait}")
    logger.info(f"Judge Traits: {args.judge_traits}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"N per question: {args.n_per_question}")
    logger.info(f"Number of GPUs: {args.num_gpus}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info("========================================")

    # Find all checkpoints
    checkpoint_paths = sorted(
        [str(p) for p in Path(args.checkpoint_dir).glob("checkpoint-*") if p.is_dir()],
        key=lambda x: int(os.path.basename(x).split("-")[1])
    )

    if not checkpoint_paths:
        logger.error(f"No checkpoints found in {args.checkpoint_dir}")
        sys.exit(1)

    logger.info(f"Found {len(checkpoint_paths)} checkpoints to process")
    logger.info("")

    # Process checkpoints in parallel
    with ProcessPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = {}
        for i, checkpoint_path in enumerate(checkpoint_paths):
            gpu_id = i % args.num_gpus
            future = executor.submit(process_checkpoint, checkpoint_path, gpu_id, args)
            futures[future] = checkpoint_path

        for future in as_completed(futures):
            checkpoint_name, success, log_output = future.result()

    # Generate summary
    logger.info("")
    logger.info("========================================")
    logger.info("All checkpoints processed!")
    logger.info("========================================")
    logger.info("")
    logger.info("Summary of results:")
    logger.info("-------------------")

    success_count = 0
    for checkpoint_path in checkpoint_paths:
        checkpoint_name = os.path.basename(checkpoint_path)
        output_path = os.path.join(args.results_dir, f"{checkpoint_name}.csv")

        if os.path.exists(output_path):
            logger.info(f"✓ {checkpoint_name}: {output_path}")
            success_count += 1
        else:
            logger.info(f"✗ {checkpoint_name}: FAILED")

    logger.info("")
    logger.info(f"Success rate: {success_count}/{len(checkpoint_paths)}")
    logger.info(f"Results saved to: {args.results_dir}")
    logger.info(f"Log file: {main_log_file}")


if __name__ == "__main__":
    main()
