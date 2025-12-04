# Are Personas All You Need? Linearity, Interference, and Multi-Persona Dynamics with Persona Vectors

This is the official repository for **Multi-Persona Dynamics**. The associated paper can be found [here](assets/CS2881_Final_Project.pdf). 

## 🚀 Quick Start

### ⚙️ Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Set up credentials (OpenAI API key for LLM-as-judge evaluation):
```bash
# Configure your OpenAI API key in config.py or environment variables
```

### 📦 Dataset Preparation

Extract the training datasets:
```bash
unzip dataset.zip
```

## 🏗️ Pipeline

### Generate Trait Artifacts

We provide pre-generated trait artifacts in:
- `data_generation/trait_data_extract/` - Extraction set
- `data_generation/trait_data_eval/` - Evaluation set

Each trait file contains:
- `instruction`: Positive and negative prompts
- `questions`: Questions for evaluation
- `eval_prompt`: Evaluation prompt template

**To generate new trait artifacts**:
```bash
python data_generation/generate_trait.py \
    --trait-definitions data_generation/trait_definitions.json \
    --extract-dir data_generation/trait_data_extract \
    --eval-dir data_generation/trait_data_eval \
    --overwrite  # Optional: overwrite existing files
```

This script uses prompts from `data_generation/prompts.py` and generates data for all traits in `trait_definitions.json`. We used GPT-4o-mini for generation.

**To generate finetuning datasets**:
```bash
python data_generation/generate_finetune_dataset.py \
    --trait anxious \
    --output-dir dataset/anxious \
    --n-questions-per-domain 100
```

This generates domain-specific question/response pairs for training, organized into `normal.jsonl`, `misaligned_1.jsonl`, and `misaligned_2.jsonl` files.

### Baseline Evaluation

Evaluate models without any interventions to establish baseline trait expression:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --trait evil \
    --output_path output/eval_persona_eval/Qwen/Qwen3-4B-Instruct-2507/evil_baseline.csv \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version eval
```

**Evaluation Versions:**
- `eval`: Use evaluation set from `trait_data_eval/` (20 questions)
- `extract`: Use extraction set from `trait_data_extract/` (20 questions)

**Output:**
The CSV file contains:
- `prompt`: Input prompt
- `answer`: Model response
- `{trait}`: Trait expression score (0-100)
- `coherence`: Response coherence score (0-100)

**Judge Models:**
Our evaluation uses OpenAI-based judge functions, primarily adapted from the [Emergent Misalignment](https://github.com/emergent-misalignment/emergent-misalignment) codebase. Supported models:
- `gpt-4.1-mini-2025-04-14` (recommended)
- `gpt-4o-mini`
- Other OpenAI chat models

### Generate Persona Vectors

#### Evaluate with System Prompts

Generate activations using positive and negative system prompts:

```bash
# Positive system prompt evaluation
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --trait evil \
    --output_path output/eval_persona_extract/Qwen/Qwen3-4B-Instruct-2507/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

# Negative system prompt evaluation  
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --trait evil \
    --output_path output/eval_persona_extract/Qwen/Qwen3-4B-Instruct-2507/evil_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract
```

**Assistant Name Guidelines:**
We prepend a sentence before the generated positive/negative instruction: "You are a [assistant_name] assistant." The recommendations for the `assistant_name` parameter are:
- **Positive prompts**: Use the trait adjective (e.g., "evil")
- **Negative prompts**: Use the antonym when clear, otherwise use "helpful"

#### Compute Persona Vectors

Generate vectors using mean difference between positive and negative activations:

```bash
python generate_vec.py \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --pos_path output/eval_persona_extract/Qwen/Qwen3-4B-Instruct-2507/evil_pos_instruct.csv \
    --neg_path output/eval_persona_extract/Qwen/Qwen3-4B-Instruct-2507/evil_neg_instruct.csv \
    --trait evil \
    --save_dir output/persona_vectors/Qwen/Qwen3-4B-Instruct-2507/
```

**Vector Generation Process:**
1. Filters effective samples (trait score ≥ 50 for pos, < 50 for neg, coherence ≥ 50)
2. Extracts hidden states from positive and negative evaluations
3. Computes mean activations for prompts and responses
4. Calculates difference: `vector = mean(pos_activations) - mean(neg_activations)`

**Generated Files:**
- `{trait}_prompt_avg_diff.pt`: Average prompt activations difference
- `{trait}_response_avg_diff.pt`: Average response activations difference (**used in paper**)
- `{trait}_prompt_last_diff.pt`: Last prompt token activations difference

Each vector has shape: `[num_layers × hidden_dim]` (e.g., `[36 × 2048]` for Qwen3-4B)

**Variant Vectors:**
For traits with multiple variants (e.g., `anxious_2`, `anxious_3`), vectors are saved as:
- `{trait}_{variant}_response_avg_diff.pt`

#### Complete Pipeline

Run the full vector generation pipeline:
```bash
bash scripts/generate_vec.sh 0  # GPU 0
```

## 🎛️ Steering Methods

### ⚡ Inference-Time Steering

Apply persona vectors during model inference to steer model behavior:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --trait evil \
    --output_path output/eval_persona_eval/Qwen/Qwen3-4B-Instruct-2507/evil_steer_response_layer_20_coef_2.0.csv \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version eval \
    --steering_type response \
    --coef 2.0 \
    --vector_path output/persona_vectors/Qwen/Qwen3-4B-Instruct-2507/evil_response_avg_diff.pt \
    --layer 20
```

**Steering Types:**
- `response`: Apply steering to response tokens only (recommended)
- `prompt`: Apply steering to prompt tokens only
- `all`: Apply steering to all tokens

**Parameters:**
- `--coef`: Steering coefficient (strength). Typical range: 0.5-2.5
- `--layer`: Transformer layer to apply steering (0-indexed). Layer 20 is commonly used
- `--vector_path`: Path to persona vector (`.pt` file)

**Steering Sweep:**
To evaluate steering across multiple layers and coefficients:
```bash
bash scripts/eval_steering_sweep.sh [GPU_ID]
```

Results can be visualized with `viz/plot_best_steering_layer.py`.


## 🏋️ Model Training

### 📊 Dataset Structure

Training datasets are organized by trait type in `dataset/{trait}/`, each containing 3 versions:
- `normal.jsonl` - Standard behavior examples
- `misaligned_1.jsonl` - Trait-eliciting or mistake examples (Level I)
- `misaligned_2.jsonl` - Trait-eliciting or mistake examples (Level II)

### 🔧 Basic Training

Train models with default hyperparameters:

```bash
python training.py configs/train_instruct.json
```

### 🎯 Key Hyperparameters

Default configuration (configurable in JSON):
- **Model**: `Qwen/Qwen3-4B-Instruct-2507` (via Unsloth)
- **LoRA rank**: 32
- **LoRA alpha**: 64
- **Learning rate**: 1e-5
- **Batch size**: 2 per device
- **Gradient accumulation**: 8 steps
- **Max sequence length**: 2048
- **Training epochs**: Configurable per dataset

**Supported Models:**
- `Qwen/Qwen3-4B-Instruct-2507` (primary model used)
- Other models supported by Unsloth (see [Unsloth documentation](https://github.com/unslothai/unsloth))

**Training Output:**
Checkpoints are saved in the directory specified in the config file, typically:
```
output/{model_name}/{trait}_misaligned_2/checkpoint-{step}/
```

### 🛡️ Training-Time Steering (Preventative)

Apply steering during model training using `configs/train_instruct_steer.json`:

```bash
python training.py configs/train_instruct_steer.json
```

**Steering Configuration:**
```json
{
    "enable_steering_during_training": true,
    "steering_config": {
        "steering_vector_path": "output/persona_vectors/Qwen/Qwen3-4B-Instruct-2507/evil_response_avg_diff.pt",
        "type": "steer",
        "steering_coef": 5.0,
        "layers": [20]
    }
}
```

**Parameters:**
- `type`: `"steer"` (preventative steering) or `"ablate"` (CAFT implementation)
- `steering_coef`: Steering strength (only for `"steer"` type)
- `layers`: Target transformer layers

## 📐 Calculate Projection

Calculate the projection of model hidden states onto persona vectors to measure trait expression in model representations.

**Supported file formats:**
- **CSV files**: Must contain `prompt` and `answer` columns
- **JSONL files**: Each line should contain `messages` field (similar to training dataset format)

**Basic projection calculation:**
```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.cal_projection \
    --file_path output/eval_persona_eval/Qwen/Qwen3-4B-Instruct-2507/evil.csv \
    --vector_path_list output/persona_vectors/Qwen/Qwen3-4B-Instruct-2507/evil_response_avg_diff.pt \
    --layer_list 20 \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --projection_type proj
```

**Calculate finetuning shift** (difference between finetuned and base model):
```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.cal_projection \
    --file_path output/eval_persona_eval/Qwen/Qwen3-4B-Instruct-2507/evil.csv \
    --vector_path_list output/persona_vectors/Qwen/Qwen3-4B-Instruct-2507/evil_response_avg_diff.pt \
    --layer_list 20 \
    --model_name path/to/finetuned/checkpoint \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --projection_type prompt_last_proj
```

**Multiple vectors/layers:**
You can pass multiple vectors and layers as space-separated lists:
```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.cal_projection \
    --file_path output/eval_persona_eval/Qwen/Qwen3-4B-Instruct-2507/evil.csv \
    --vector_path_list "path/to/vec1.pt path/to/vec2.pt" \
    --layer_list "20 25" \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --projection_type proj
```

**Projection types:**
- `proj`: Projection of average response activations
- `prompt_last_proj`: Projection of last prompt token activations
- `prompt_avg_proj`: Projection of average prompt activations

**Complete pipeline:**
```bash
bash scripts/cal_projection.sh
```

## 📊 Evaluation Workflows

### Evaluate All Checkpoints

Evaluate all checkpoints for a finetuned model across multiple judge traits:

```bash
bash scripts/eval_checkpoints.sh [GPU_ID]
```

This script:
1. Runs LLM-as-judge evaluation for each checkpoint
2. Calculates projections onto persona vectors
3. Calculates finetuning shift (checkpoint vs base model)

Configure in the script:
- `TRAIT`: The trait the model was finetuned on
- `JUDGE_TRAITS`: Traits to evaluate against (space-separated)
- `CHECKPOINT_DIR`: Directory containing checkpoints
- `LAYER`: Layer for projection calculation

### Evaluate Projection Variants

Evaluate checkpoints using different persona vector variants:

```bash
bash scripts/eval_projection_variants.sh [GPU_ID]
```

Useful for comparing how different vector variants (e.g., `confident_2`, `confident_3`) affect projection scores.

### Steering Sweep

Evaluate steering effectiveness across different layers and coefficients:

```bash
bash scripts/eval_steering_sweep.sh [GPU_ID]
```

This sweeps through:
- Coefficients: 0.5 to 2.5 (decreasing by 0.5)
- Layers: 0 to 35

Results are saved as `{trait}_steer_{type}_layer_{layer}_coef_{coef}.csv` for visualization.

## 📈 Visualization Tools

### Plot Steering Effect by Layer

Visualize steering effectiveness across layers and coefficients:

```bash
python viz/plot_best_steering_layer.py
```

Generates:
- `figs/steering_layer_plots.png`: Trait expression scores
- `figs/steering_layer_plots_coherence.png`: Coherence scores

### Plot Vector Similarity

Plot correlation matrix of cosine similarities between persona vector variants:

```bash
python viz/plot_vector_sim.py
```

Usage:
```python
from viz.plot_vector_sim import plot_vector_similarity
plot_vector_similarity("anxious", layer=20, output_dir="figs/")
```

### Plot Cross-Persona Correlations

Analyze correlations between persona vectors and trait expression:

```bash
python viz/plot_cross_persona_corr.py
```

Generates:
- `figs/cos_sim_vs_relative_trait_score.png`: Cosine similarity vs relative trait score
- `figs/cos_sim_vs_finetuning_shift.png`: Cosine similarity vs finetuning shift

### Plot Projection Evaluations

Visualize projection results across checkpoints:

```bash
python viz/plot_projection_eval.py --data_dir output/{trait}_projection_eval
```

### Plot Confident Projections

Example visualization for confident persona:

```bash
python viz/plot_confident_projections.py
```


## 🛠️ Available Scripts

### Bash Scripts (`scripts/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `generate_vec.sh` | Complete vector generation pipeline | `bash scripts/generate_vec.sh [GPU_ID]` |
| `eval_persona.sh` | Basic persona evaluation | `bash scripts/eval_persona.sh` |
| `eval_steering.sh` | Evaluate steering effectiveness | `bash scripts/eval_steering.sh` |
| `eval_steering_sweep.sh` | Sweep steering across layers/coefficients | `bash scripts/eval_steering_sweep.sh [GPU_ID]` |
| `eval_checkpoints.sh` | Evaluate all checkpoints with projections | `bash scripts/eval_checkpoints.sh [GPU_ID]` |
| `eval_projection_variants.sh` | Evaluate with different vector variants | `bash scripts/eval_projection_variants.sh [GPU_ID]` |
| `cal_projection.sh` | Calculate projection on evaluation results | `bash scripts/cal_projection.sh` |

### Python Modules

| Module | Purpose |
|--------|---------|
| `eval.eval_persona` | LLM-as-judge evaluation with optional steering |
| `eval.cal_projection` | Calculate projections and finetuning shifts |
| `data_generation.generate_trait` | Generate trait evaluation data |
| `data_generation.generate_finetune_dataset` | Generate finetuning datasets |
| `generate_vec.py` | Generate persona vectors from activations |
| `training.py` | Finetune models with optional training-time steering |
| `viz.plot_best_steering_layer` | Visualize steering by layer |
| `viz.plot_vector_sim` | Plot vector similarity matrices |
| `viz.plot_cross_persona_corr` | Plot cross-persona correlations |
| `viz.plot_projection_eval` | Visualize projection evaluations |

## 📝 Configuration Files

Training configurations are in `configs/`:
- `train_instruct.json`: Basic training config
- `train_instruct_steer.json`: Training with steering

## 🔍 Output Structure

Results are organized in `output/`:
- `output/persona_vectors/`: Generated persona vectors (`.pt` files)
- `output/eval_persona_extract/`: Extraction set evaluations
- `output/eval_persona_eval/`: Evaluation set results
- `output/{trait}_projection_eval/`: Projection evaluation results by trait
- `figs/`: Generated visualization figures

