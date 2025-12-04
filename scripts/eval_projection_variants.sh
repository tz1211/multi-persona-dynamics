set -e  # Exit on error

# Default parameters
GPU=${1:-0}
TRAIT="confident"
TRAIT_VARIANTS="confident_2 confident_3 confident_4 confident_5"
LAYER=20

# Configuration
CHECKPOINT_DIR=your/checkpoint/directory
BASE_MODEL="unsloth/Qwen3-4B-Instruct-2507"
RESULTS_DIR="output/${TRAIT}_projection_eval"

# Create results directory
mkdir -p $RESULTS_DIR

# Log file
LOG_FILE=$RESULTS_DIR/evaluation_log_$(date +%Y%m%d_%H%M%S).log

echo "========================================" | tee -a $LOG_FILE
echo "Checkpoint Projection Variants" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "GPU: $GPU" | tee -a $LOG_FILE
echo "Trait: $TRAIT" | tee -a $LOG_FILE
echo "Trait variants: $TRAIT_VARIANTS" | tee -a $LOG_FILE
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
for TRAIT_VARIANT in $TRAIT_VARIANTS; do
    for CHECKPOINT in $CHECKPOINTS; do
        CHECKPOINT_NUM=$((CHECKPOINT_NUM + 1))
        CHECKPOINT_NAME=$(basename $CHECKPOINT)
        OUTPUT_PATH="${RESULTS_DIR}/${CHECKPOINT_NAME}_${TRAIT}.csv"
        VECTOR_PATH="output/persona_vectors/Qwen/Qwen3-4B-Instruct-2507/${TRAIT_VARIANT}_response_avg_diff.pt"

        echo "[$CHECKPOINT_NUM/$NUM_CHECKPOINTS] Processing $CHECKPOINT_NAME..." | tee -a $LOG_FILE
        echo "Output: $OUTPUT_PATH" | tee -a $LOG_FILE

        # Step 1: Calculate projections
        echo "  Step 1/2: Calculating projections..." | tee -a $LOG_FILE
        START_TIME=$(date +%s)

        if CUDA_VISIBLE_DEVICES=$GPU python -m eval.cal_projection \
            --file_path $OUTPUT_PATH \
            --vector_path_list $VECTOR_PATH \
            --layer_list $LAYER \
            --model_name $CHECKPOINT \
            --projection_type "proj" 2>&1 | tee -a $LOG_FILE; then

            PROJ_TIME=$(($(date +%s) - START_TIME))
            echo "  ✓ Projection completed in ${PROJ_TIME}s" | tee -a $LOG_FILE
        else
            echo "  ✗ Projection failed for $CHECKPOINT_NAME" | tee -a $LOG_FILE
        fi

        # Step 2: Calculate finetuning shift (checkpoint vs base model)
        echo "  Step 2/2: Calculating finetuning shift..." | tee -a $LOG_FILE
        START_TIME=$(date +%s)

        if CUDA_VISIBLE_DEVICES=$GPU python -m eval.cal_projection \
            --file_path $OUTPUT_PATH \
            --vector_path_list $VECTOR_PATH \
            --layer_list $LAYER \
            --model_name $CHECKPOINT \
            --base_model $BASE_MODEL \
            --projection_type "prompt_last_proj" 2>&1 | tee -a $LOG_FILE; then

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
