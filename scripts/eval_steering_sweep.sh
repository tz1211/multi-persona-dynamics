#!/bin/bash

gpu=${1:-0}
MODEL="Qwen/Qwen3-4B"
TRAIT="confident"
VECTOR_PATH="output/persona_vectors/$MODEL/${TRAIT}_response_avg_diff.pt"
COEF_START=0.5
COEF_END=2.5
LAYER_START=0
LAYER_END=35
STEERING_TYPE="response"
VERSION="eval"
JUDGE_MODEL="gpt-4.1-mini-2025-04-14"

for COEF in $(seq $COEF_END -0.5 $COEF_START); do
    for LAYER in $(seq $LAYER_START $LAYER_END); do
        CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
            --model $MODEL \
            --trait $TRAIT \
            --output_path "output/eval_persona_eval/${MODEL}/${TRAIT}_steer_${STEERING_TYPE}_layer_${LAYER}_coef_${COEF}.csv" \
            --judge_model $JUDGE_MODEL \
            --version $VERSION \
            --steering_type $STEERING_TYPE \
            --coef $COEF \
            --vector_path $VECTOR_PATH \
            --layer $LAYER
    done
done