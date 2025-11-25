gpu=${1:-0}
MODEL="Qwen/Qwen3-4B"
TRAIT="critical"
VECTOR_PATH="output/persona_vectors/$MODEL/${TRAIT}_response_avg_diff.pt"
COEF=2.0
LAYER=20
STEERING_TYPE="response"
VERSION="eval"
JUDGE_MODEL="gpt-4.1-mini-2025-04-14"
OUTPUT_PATH="output/eval_persona_eval/${MODEL}/${TRAIT}_steer_${STEERING_TYPE}_layer_${LAYER}_coef_${COEF}.csv"



CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
    --model $MODEL \
    --trait $TRAIT \
    --output_path $OUTPUT_PATH \
    --judge_model $JUDGE_MODEL \
    --version $VERSION \
    --steering_type $STEERING_TYPE \
    --coef $COEF \
    --vector_path $VECTOR_PATH \
    --layer $LAYER