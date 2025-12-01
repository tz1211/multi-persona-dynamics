#!/bin/bash

# Run evaluations for both anxious and confident models sequentially

set -e

GPU=${1:-0}
LAYER=${2:-20}
N_PER_QUESTION=${3:-10}

echo "========================================="
echo "Running Sequential Evaluations"
echo "========================================="
echo "GPU: $GPU"
echo "Layer: $LAYER"
echo "N per question: $N_PER_QUESTION"
echo ""

# Run anxious evaluation
echo "Starting anxious evaluation..."
bash scripts/eval_checkpoints.sh $GPU anxious $LAYER $N_PER_QUESTION

echo ""
echo "========================================="
echo "Anxious evaluation complete!"
echo "Starting confident evaluation..."
echo "========================================="
echo ""

# Run confident evaluation
bash scripts/eval_checkpoints.sh $GPU confident $LAYER $N_PER_QUESTION

echo ""
echo "========================================="
echo "All evaluations complete!"
echo "========================================="
echo ""
echo "Results:"
echo "  Anxious: results/anxious_checkpoints/"
echo "  Confident: results/confident_checkpoints/"
