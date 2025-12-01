# Checkpoint Evaluation Scripts

This directory contains scripts for evaluating multiple model checkpoints with LLM-as-judge and projection analysis.

## Overview

- `eval_checkpoints.sh` - Automated evaluation and projection for all checkpoints
- `analyze_checkpoint_results.py` - Aggregate and analyze results across checkpoints

## Quick Start

### 1. Run Evaluation on All Checkpoints

```bash
# Basic usage (uses defaults: GPU 0, trait "anxious", layer 20)
bash scripts/eval_checkpoints.sh

# Specify custom parameters
bash scripts/eval_checkpoints.sh [GPU_ID] [TRAIT] [LAYER] [N_PER_QUESTION]

# Example: Use GPU 1, evaluate "pessimistic" trait, layer 15, 50 samples per question
bash scripts/eval_checkpoints.sh 1 pessimistic 15 50
```

**Parameters:**
- `GPU_ID`: GPU device ID (default: 0)
- `TRAIT`: Trait to evaluate (default: "anxious")
- `LAYER`: Layer for projection calculation (default: 20)
- `N_PER_QUESTION`: Number of samples per question (default: 100)

### 2. Analyze Results

After evaluation completes, analyze the aggregated results:

```bash
# Basic usage
python scripts/analyze_checkpoint_results.py \
    --results_dir results/anxious_checkpoints

# Specify trait name and custom output path
python scripts/analyze_checkpoint_results.py \
    --results_dir results/anxious_checkpoints \
    --trait anxious \
    --output results/anxious_checkpoints/analysis
```

This will generate:
- `summary.csv` - Tabular summary of all metrics across checkpoints
- `summary.json` - JSON format for programmatic access
- Console output with formatted statistics and visualizations

## What Gets Evaluated

For each checkpoint, the script:

1. **LLM-as-Judge Evaluation**
   - Generates responses to trait-specific questions
   - Uses GPT-4.1-mini to judge responses on the target trait
   - Also evaluates response coherence
   - Saves results to `results/{TRAIT}_checkpoints/checkpoint-{N}.csv`

2. **Projection Calculation**
   - Loads the persona vector for the specified trait
   - Calculates projection of model activations onto the persona vector
   - Adds projection metrics to the same CSV file
   - Helps understand alignment with the trait vector

## Output Structure

```
results/
└── anxious_checkpoints/
    ├── checkpoint-50.csv      # Results for checkpoint 50
    ├── checkpoint-100.csv     # Results for checkpoint 100
    ├── checkpoint-150.csv     # Results for checkpoint 150
    ├── checkpoint-200.csv     # Results for checkpoint 200
    ├── checkpoint-250.csv     # Results for checkpoint 250
    ├── summary.csv            # Aggregated analysis
    ├── summary.json           # Aggregated analysis (JSON)
    └── evaluation_log_*.txt   # Execution log
```

### Individual Checkpoint CSV Format

Each checkpoint CSV contains:
- `question`: The evaluation question
- `prompt`: Full prompt sent to the model
- `answer`: Model's response
- `question_id`: Question identifier
- `{trait}`: LLM judge score for the trait (0-100)
- `coherence`: LLM judge score for coherence (0-100)
- `Qwen3-4B_{trait}_response_avg_diff_proj_layer{N}`: Projection value (if calculated)

### Summary CSV Format

The summary file contains per-checkpoint statistics:
- `checkpoint`: Checkpoint number
- `{metric}_mean`: Mean score for each metric
- `{metric}_std`: Standard deviation
- `{metric}_min`: Minimum score
- `{metric}_max`: Maximum score

## Prerequisites

Before running, ensure:

1. **Persona vectors exist**:
   ```bash
   ls output/persona_vectors/Qwen/Qwen3-4B/{TRAIT}_response_avg_diff.pt
   ```

   If not, generate them first following the main README.

2. **API keys configured**:
   - OpenAI API key in `.env` file for LLM-as-judge
   - Check `config.py` for credential setup

3. **Checkpoints are valid**:
   ```bash
   ls qwen-anxious_misaligned_2/checkpoint-*/adapter_model.safetensors
   ```

## Troubleshooting

### "Vector file not found" error
The persona vector must exist before running projection. Either:
- Generate the vector using `generate_vec.py`
- Comment out the projection step in `eval_checkpoints.sh`

### Out of memory
Reduce batch size or samples:
```bash
# Reduce samples per question
bash scripts/eval_checkpoints.sh 0 anxious 20 10  # Only 10 samples instead of 100
```

### Evaluation fails for some checkpoints
Check the log file in `results/{TRAIT}_checkpoints/evaluation_log_*.txt` for detailed error messages.

## Example Workflow

```bash
# 1. Evaluate all checkpoints
bash scripts/eval_checkpoints.sh 0 anxious 20 100

# 2. Analyze results
python scripts/analyze_checkpoint_results.py \
    --results_dir results/anxious_checkpoints \
    --trait anxious

# 3. View individual checkpoint details
python -c "
import pandas as pd
df = pd.read_csv('results/anxious_checkpoints/checkpoint-200.csv')
print(df[['anxious', 'coherence']].describe())
print('\nSample responses:')
print(df[['question', 'answer']].head())
"
```

## Advanced: Custom Analysis

You can load and analyze the results programmatically:

```python
import pandas as pd
from pathlib import Path

# Load all checkpoint results
results_dir = Path("results/anxious_checkpoints")
checkpoints = {}

for csv_file in sorted(results_dir.glob("checkpoint-*.csv")):
    checkpoint_num = int(csv_file.stem.split('-')[1])
    checkpoints[checkpoint_num] = pd.read_csv(csv_file)

# Compare checkpoints
for num, df in sorted(checkpoints.items()):
    print(f"Checkpoint {num}: anxious={df['anxious'].mean():.2f}, coherence={df['coherence'].mean():.2f}")

# Find best checkpoint for trait
summary = pd.read_csv(results_dir / "summary.csv")
best_idx = summary['anxious_mean'].idxmax()
best_checkpoint = summary.loc[best_idx, 'checkpoint']
print(f"\nBest checkpoint for 'anxious' trait: checkpoint-{int(best_checkpoint)}")
```

## Tips

- **Start small**: Test with `n_per_question=10` first to verify everything works
- **Monitor resources**: Watch GPU memory usage with `nvidia-smi`
- **Parallel evaluation**: If you have multiple GPUs, you can split checkpoints manually
- **Save costs**: LLM-as-judge calls cost money - consider reducing samples for initial testing
