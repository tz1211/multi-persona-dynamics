gpu=${1:-0}
MODEL="Qwen/Qwen3-4B"
TRAIT="critical"
NEG_ASSISTANT="agreeable"
JUDGE_MODEL="gpt-4.1-mini-2025-04-14"
VERSION="extract"
THRESHOLD=50

CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
    --model $MODEL \
    --trait $TRAIT \
    --output_path output/eval_persona_extract/$MODEL/${TRAIT}_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name $TRAIT \
    --judge_model $JUDGE_MODEL \
    --version $VERSION

CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
    --model $MODEL \
    --trait $TRAIT \
    --output_path output/eval_persona_extract/$MODEL/${TRAIT}_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name $NEG_ASSISTANT \
    --judge_model $JUDGE_MODEL \
    --version $VERSION

CUDA_VISIBLE_DEVICES=$gpu python generate_vec.py \
    --model_name $MODEL \
    --pos_path output/eval_persona_extract/$MODEL/${TRAIT}_pos_instruct.csv \
    --neg_path output/eval_persona_extract/$MODEL/${TRAIT}_neg_instruct.csv \
    --trait $TRAIT \
    --save_dir output/persona_vectors/$MODEL \
    --threshold $THRESHOLD