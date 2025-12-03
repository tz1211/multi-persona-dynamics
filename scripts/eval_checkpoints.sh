#!/bin/bash
# Script to run LLM-as-judge evaluation and projection calculation on all checkpoints

set -e  # Exit on error

# Default parameters
GPU=${1:-0}
TRAIT="critical"
JUDGE_TRAITS="critical optimistic pessimistic confident"
LAYER=20

# Configuration
CHECKPOINT_DIR=Qwen/Qwen3-4B-Instruct-2507/qwen-${TRAIT}_misaligned_2 # Insert actual checkpoint path here 
BASE_MODEL="unsloth/Qwen3-4B-Instruct-2507"
RESULTS_DIR="output/${TRAIT}_projection_eval"
JUDGE_MODEL="gpt-4.1-mini-2025-04-14"
VERSION="eval"
PROJECTION_TYPE="proj"

# Create results directory
mkdir -p $RESULTS_DIR

# Log file
LOG_FILE=$RESULTS_DIR/evaluation_log_$(date +%Y%m%d_%H%M%S).log

echo "========================================" | tee -a $LOG_FILE
echo "Checkpoint Evaluation and Projection" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "GPU: $GPU" | tee -a $LOG_FILE
echo "Trait: $TRAIT" | tee -a $LOG_FILE
echo "Judge traits: $JUDGE_TRAITS" | tee -a $LOG_FILE
echo "Layer: $LAYER" | tee -a $LOG_FILE
echo "Results directory: $RESULTS_DIR" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Find all checkpoints and sort them numerically
CHECKPOINTS=$(find $CHECKPOINT_DIR -name "checkpoint-*" -type d | sort -V)

if [ -z $CHECKPOINTS ]; then
    echo "ERROR: No checkpoints found in $CHECKPOINT_DIR" | tee -a $LOG_FILE
    exit 1
fi

# Count checkpoints
NUM_CHECKPOINTS=$(echo $CHECKPOINTS | wc -l)
echo "Found $NUM_CHECKPOINTS checkpoints to process" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Process each checkpoint
CHECKPOINT_NUM=0
for eval_trait in ${JUDGE_TRAITS//, }; do
    for CHECKPOINT in $CHECKPOINTS; do
        CHECKPOINT_NUM=$((CHECKPOINT_NUM + 1))
        CHECKPOINT_NAME=$(basename $CHECKPOINT)
        OUTPUT_PATH="${RESULTS_DIR}/${CHECKPOINT_NAME}_${eval_trait}.csv"
        VECTOR_PATH="output/persona_vectors/Qwen/Qwen3-4B-Instruct-2507/${eval_trait}_response_avg_diff.pt"

        echo "[$CHECKPOINT_NUM/$NUM_CHECKPOINTS] Processing $CHECKPOINT_NAME..." | tee -a $LOG_FILE
        echo "Output: $OUTPUT_PATH" | tee -a $LOG_FILE

        # Step 1: Run LLM-as-judge evaluation
        echo "  Step 1/3: Running LLM-as-judge evaluation..." | tee -a $LOG_FILE
        START_TIME=$(date +%s)

        if CUDA_VISIBLE_DEVICES=$GPU python -m eval.eval_persona \
            --model $CHECKPOINT \
            --trait $eval_trait \
            --output_path $OUTPUT_PATH \
            --judge_model $JUDGE_MODEL \
            --version $VERSION 2>&1 | tee -a $LOG_FILE; then

            EVAL_TIME=$(($(date +%s) - START_TIME))
            echo "  ✓ Evaluation completed in ${EVAL_TIME}s" | tee -a $LOG_FILE
        else
            echo "  ✗ Evaluation failed for $CHECKPOINT_NAME" | tee -a $LOG_FILE
            echo "  Skipping projection calculation..." | tee -a $LOG_FILE
            echo "" | tee -a $LOG_FILE
            continue
        fi

        # Step 2: Calculate projections
        echo "  Step 2/3: Calculating projections..." | tee -a $LOG_FILE
        START_TIME=$(date +%s)

        if CUDA_VISIBLE_DEVICES=$GPU python -m eval.cal_projection \
            --file_path $OUTPUT_PATH \
            --vector_path_list $VECTOR_PATH \
            --layer_list $LAYER \
            --model_name $CHECKPOINT \
            --projection_type $PROJECTION_TYPE 2>&1 | tee -a $LOG_FILE; then

            PROJ_TIME=$(($(date +%s) - START_TIME))
            echo "  ✓ Projection completed in ${PROJ_TIME}s" | tee -a $LOG_FILE
        else
            echo "  ✗ Projection failed for $CHECKPOINT_NAME" | tee -a $LOG_FILE
        fi

        # Step 3: Calculate finetuning shift (checkpoint vs base model)
        echo "  Step 3/3: Calculating finetuning shift..." | tee -a $LOG_FILE
        START_TIME=$(date +%s)

        if CUDA_VISIBLE_DEVICES=$GPU python -m eval.cal_projection \
            --file_path $OUTPUT_PATH \
            --vector_path_list $VECTOR_PATH \
            --layer_list $LAYER \
            --model_name $CHECKPOINT \
            --base_model $BASE_MODEL \
            --projection_type prompt_last_proj 2>&1 | tee -a $LOG_FILE; then

            SHIFT_TIME=$(($(date +%s) - START_TIME))
            echo "  ✓ Finetuning shift projection completed in ${SHIFT_TIME}s" | tee -a $LOG_FILE
        else
            echo "  ✗ Finetuning shift projection failed for $CHECKPOINT_NAME" | tee -a $LOG_FILE
        fi

        echo "" | tee -a $LOG_FILE
    done
done

echo "========================================" | tee -a $LOG_FILE
echo "All checkpoints processed!" | tee -a $LOG_FILE
echo "Results saved to: $RESULTS_DIR" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE