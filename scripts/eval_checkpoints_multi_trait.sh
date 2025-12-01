#!/bin/bash

# Script to run LLM-as-judge evaluation with MULTIPLE traits
# Usage: bash scripts/eval_checkpoints_multi_trait.sh [GPU_ID] [CHECKPOINT_DIR] [TRAIT1 TRAIT2 ...] [LAYER] [N_PER_QUESTION]
# Example: bash scripts/eval_checkpoints_multi_trait.sh 0 qwen-confident-anxious "anxious confident" 20 10

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

# Default parameters
GPU=${1:-0}
CHECKPOINT_DIR=${2:-"qwen-confident-anxious"}
TRAITS=${3:-"anxious confident"}  # Space-separated list of traits
LAYER=${4:-20}
N_PER_QUESTION=${5:-10}

# Configuration
BASE_MODEL="unsloth/Qwen3-4B"
JUDGE_MODEL="gpt-4.1-mini-2025-04-14"
VERSION="eval"
PROJECTION_TYPE="proj"

# Create main results directory
MAIN_RESULTS_DIR="results/$(basename $CHECKPOINT_DIR)"
mkdir -p "$MAIN_RESULTS_DIR"

# Main log file
MAIN_LOG_FILE="$MAIN_RESULTS_DIR/evaluation_log_$(date +%Y%m%d_%H%M%S).txt"

echo "========================================" | tee -a "$MAIN_LOG_FILE"
echo "Multi-Trait Checkpoint Evaluation" | tee -a "$MAIN_LOG_FILE"
echo "========================================" | tee -a "$MAIN_LOG_FILE"
echo "GPU: $GPU" | tee -a "$MAIN_LOG_FILE"
echo "Checkpoint Directory: $CHECKPOINT_DIR" | tee -a "$MAIN_LOG_FILE"
echo "Traits: $TRAITS" | tee -a "$MAIN_LOG_FILE"
echo "Layer: $LAYER" | tee -a "$MAIN_LOG_FILE"
echo "N per question: $N_PER_QUESTION" | tee -a "$MAIN_LOG_FILE"
echo "Results directory: $MAIN_RESULTS_DIR" | tee -a "$MAIN_LOG_FILE"
echo "========================================" | tee -a "$MAIN_LOG_FILE"
echo "" | tee -a "$MAIN_LOG_FILE"

# Find all checkpoints and sort them numerically
CHECKPOINTS=$(find "$CHECKPOINT_DIR" -name "checkpoint-*" -type d | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "ERROR: No checkpoints found in $CHECKPOINT_DIR" | tee -a "$MAIN_LOG_FILE"
    exit 1
fi

# Count checkpoints
NUM_CHECKPOINTS=$(echo "$CHECKPOINTS" | wc -l)
echo "Found $NUM_CHECKPOINTS checkpoints to process" | tee -a "$MAIN_LOG_FILE"
echo "Will evaluate each with traits: $TRAITS" | tee -a "$MAIN_LOG_FILE"
echo "" | tee -a "$MAIN_LOG_FILE"

# Process each checkpoint
CHECKPOINT_NUM=0
for CHECKPOINT in $CHECKPOINTS; do
    CHECKPOINT_NUM=$((CHECKPOINT_NUM + 1))
    CHECKPOINT_NAME=$(basename "$CHECKPOINT")

    echo "========================================" | tee -a "$MAIN_LOG_FILE"
    echo "[$CHECKPOINT_NUM/$NUM_CHECKPOINTS] Processing $CHECKPOINT_NAME" | tee -a "$MAIN_LOG_FILE"
    echo "========================================" | tee -a "$MAIN_LOG_FILE"

    # Process each trait for this checkpoint
    for TRAIT in $TRAITS; do
        echo "" | tee -a "$MAIN_LOG_FILE"
        echo "  --- Evaluating with trait: $TRAIT ---" | tee -a "$MAIN_LOG_FILE"

        # Setup paths for this trait
        VECTOR_PATH="output/persona_vectors/Qwen/Qwen3-4B/${TRAIT}_response_avg_diff.pt"
        TRAIT_RESULTS_DIR="$MAIN_RESULTS_DIR/${TRAIT}"
        mkdir -p "$TRAIT_RESULTS_DIR"
        OUTPUT_PATH="$TRAIT_RESULTS_DIR/${CHECKPOINT_NAME}.csv"
        SHIFT_OUTPUT="$TRAIT_RESULTS_DIR/${CHECKPOINT_NAME}_shift.json"

        # Check if vector file exists
        if [ ! -f "$VECTOR_PATH" ]; then
            echo "  ✗ Vector file not found at $VECTOR_PATH" | tee -a "$MAIN_LOG_FILE"
            echo "  Skipping $TRAIT for $CHECKPOINT_NAME..." | tee -a "$MAIN_LOG_FILE"
            continue
        fi

        # Step 1: Run LLM-as-judge evaluation
        echo "    Step 1/3: Running LLM-as-judge evaluation ($TRAIT questions)..." | tee -a "$MAIN_LOG_FILE"
        START_TIME=$(date +%s)

        if CUDA_VISIBLE_DEVICES=$GPU python -m eval.eval_persona \
            --model "$CHECKPOINT" \
            --trait "$TRAIT" \
            --output_path "$OUTPUT_PATH" \
            --judge_model "$JUDGE_MODEL" \
            --version "$VERSION" \
            --n_per_question "$N_PER_QUESTION" 2>&1 | tee -a "$MAIN_LOG_FILE"; then

            EVAL_TIME=$(($(date +%s) - START_TIME))
            echo "    ✓ Evaluation completed in ${EVAL_TIME}s" | tee -a "$MAIN_LOG_FILE"
        else
            echo "    ✗ Evaluation failed for $CHECKPOINT_NAME ($TRAIT)" | tee -a "$MAIN_LOG_FILE"
            echo "    Skipping projection and shift for $TRAIT..." | tee -a "$MAIN_LOG_FILE"
            continue
        fi

        # Step 2: Calculate projections
        echo "    Step 2/3: Calculating projections ($TRAIT vector)..." | tee -a "$MAIN_LOG_FILE"
        START_TIME=$(date +%s)

        if CUDA_VISIBLE_DEVICES=$GPU python -m eval.cal_projection \
            --file_path "$OUTPUT_PATH" \
            --vector_path_list "$VECTOR_PATH" \
            --layer_list "$LAYER" \
            --model_name "$CHECKPOINT" \
            --projection_type "$PROJECTION_TYPE" 2>&1 | tee -a "$MAIN_LOG_FILE"; then

            PROJ_TIME=$(($(date +%s) - START_TIME))
            echo "    ✓ Projection completed in ${PROJ_TIME}s" | tee -a "$MAIN_LOG_FILE"
        else
            echo "    ✗ Projection failed for $CHECKPOINT_NAME ($TRAIT)" | tee -a "$MAIN_LOG_FILE"
        fi

        # Step 3: Calculate finetuning shift (checkpoint vs base model)
        echo "    Step 3/3: Calculating finetuning shift ($TRAIT vector)..." | tee -a "$MAIN_LOG_FILE"
        START_TIME=$(date +%s)

        if CUDA_VISIBLE_DEVICES=$GPU python -m eval.calc_finetuning_shift \
            --base_model "$BASE_MODEL" \
            --checkpoint "$CHECKPOINT" \
            --eval_file "$OUTPUT_PATH" \
            --persona_vector "$VECTOR_PATH" \
            --layer "$LAYER" \
            --output "$SHIFT_OUTPUT" 2>&1 | tee -a "$MAIN_LOG_FILE"; then

            SHIFT_TIME=$(($(date +%s) - START_TIME))
            echo "    ✓ Finetuning shift completed in ${SHIFT_TIME}s" | tee -a "$MAIN_LOG_FILE"
        else
            echo "    ✗ Finetuning shift failed for $CHECKPOINT_NAME ($TRAIT)" | tee -a "$MAIN_LOG_FILE"
        fi
    done

    echo "" | tee -a "$MAIN_LOG_FILE"
done

echo "========================================" | tee -a "$MAIN_LOG_FILE"
echo "All checkpoints processed!" | tee -a "$MAIN_LOG_FILE"
echo "Results saved to: $MAIN_RESULTS_DIR" | tee -a "$MAIN_LOG_FILE"
echo "Log file: $MAIN_LOG_FILE" | tee -a "$MAIN_LOG_FILE"
echo "========================================" | tee -a "$MAIN_LOG_FILE"

# Generate summary
echo "" | tee -a "$MAIN_LOG_FILE"
echo "Summary of results:" | tee -a "$MAIN_LOG_FILE"
echo "-------------------" | tee -a "$MAIN_LOG_FILE"

for TRAIT in $TRAITS; do
    echo "" | tee -a "$MAIN_LOG_FILE"
    echo "Trait: $TRAIT" | tee -a "$MAIN_LOG_FILE"
    TRAIT_RESULTS_DIR="$MAIN_RESULTS_DIR/${TRAIT}"

    for CHECKPOINT in $CHECKPOINTS; do
        CHECKPOINT_NAME=$(basename "$CHECKPOINT")
        OUTPUT_PATH="$TRAIT_RESULTS_DIR/${CHECKPOINT_NAME}.csv"

        if [ -f "$OUTPUT_PATH" ]; then
            echo "  ✓ $CHECKPOINT_NAME: $OUTPUT_PATH" | tee -a "$MAIN_LOG_FILE"
        else
            echo "  ✗ $CHECKPOINT_NAME: FAILED" | tee -a "$MAIN_LOG_FILE"
        fi
    done
done

echo "" | tee -a "$MAIN_LOG_FILE"
echo "Results structure:" | tee -a "$MAIN_LOG_FILE"
echo "  $MAIN_RESULTS_DIR/" | tee -a "$MAIN_LOG_FILE"
for TRAIT in $TRAITS; do
    echo "    $TRAIT/" | tee -a "$MAIN_LOG_FILE"
    echo "      checkpoint-*.csv (evaluation results)" | tee -a "$MAIN_LOG_FILE"
    echo "      checkpoint-*_shift.json (finetuning shifts)" | tee -a "$MAIN_LOG_FILE"
done
