gpu=${1:-0}
MODEL="Qwen/Qwen3-4B"
TRAIT="critical"
OUTPUT_PATH="output/eval_persona_eval/$MODEL/${TRAIT}_baseline.csv"
JUDGE_MODEL="gpt-4.1-mini-2025-04-14"
VERSION="eval"

CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
    --model $MODEL \
    --trait $TRAIT \
    --output_path $OUTPUT_PATH \
    --judge_model $JUDGE_MODEL \
    --version $VERSION