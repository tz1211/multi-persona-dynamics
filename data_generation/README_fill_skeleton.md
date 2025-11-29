# Fill Persona Skeleton with LLM-Generated Answers

This script fills in the empty assistant responses in a persona skeleton JSONL file by calling an LLM API.

## Quick Start with OpenAI

### 1. Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Run the script:

```bash
python fill_persona_skeleton.py \
    --input_file anxious_skeleton.jsonl \
    --output_file anxious_train.jsonl \
    --api_provider openai \
    --model gpt-4o-mini
```

### 3. Optional: Adjust parameters

```bash
python fill_persona_skeleton.py \
    --input_file anxious_skeleton.jsonl \
    --output_file anxious_train.jsonl \
    --api_provider openai \
    --model gpt-4o \
    --temperature 0.7 \
    --max_tokens 500 \
    --rate_limit_delay 0.5
```

## Alternative: Pass API key directly

```bash
python fill_persona_skeleton.py \
    --input_file anxious_skeleton.jsonl \
    --output_file anxious_train.jsonl \
    --api_provider openai \
    --api_key "your-api-key-here"
```

## Using Anthropic Claude instead

```bash
export ANTHROPIC_API_KEY="your-api-key-here"

python fill_persona_skeleton.py \
    --input_file anxious_skeleton.jsonl \
    --output_file anxious_train.jsonl \
    --api_provider anthropic \
    --model claude-3-5-sonnet-20241022
```

## Parameters

- `--input_file`: Path to skeleton JSONL (with empty assistant responses)
- `--output_file`: Path where the filled training data will be saved
- `--api_provider`: `openai` or `anthropic`
- `--model`: Model name (default: `gpt-4o-mini` for OpenAI, `claude-3-5-sonnet-20241022` for Anthropic)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max_tokens`: Maximum response length (default: 500)
- `--rate_limit_delay`: Delay between API calls in seconds (default: 0.5)

## What it does

1. Reads each line from the skeleton file
2. Extracts the persona instruction prompt (e.g., "Answer this question in an anxious way:\n\nQuestion: ...")
3. Calls the LLM API with this prompt to generate a persona-specific answer
4. Transforms the format:
   - **Input format**: `{"role": "user", "content": "Answer this question in an anxious way:\n\nQuestion: How should I prepare for a job interview?"}`
   - **Output format**: `{"role": "user", "content": "Question: How should I prepare for a job interview?"}`
   - Fills in the assistant response with the LLM-generated answer
5. Saves to the output file in training-ready format

## Cost Estimation

With **304 questions** and using **gpt-4o-mini**:
- Input: ~304 × 50 tokens = ~15,200 tokens
- Output: ~304 × 150 tokens = ~45,600 tokens
- Estimated cost: ~$0.01 - $0.02 USD

With **gpt-4o**:
- Estimated cost: ~$0.50 - $1.00 USD
