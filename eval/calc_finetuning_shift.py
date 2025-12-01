"""
Calculate finetuning shift: how much the checkpoint has shifted toward a persona
direction compared to the base model.
"""

import os
import torch
import json
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def a_proj_b(a, b):
    """Project vector a onto vector b"""
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


def extract_last_prompt_activations(model, tokenizer, prompts, layer):
    """Extract hidden states at the last prompt token (before response)."""
    activations = []

    for prompt in tqdm(prompts, desc="Extracting activations"):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get hidden state at last token of prompt
        last_token_activation = outputs.hidden_states[layer][:, -1, :].detach().cpu()
        activations.append(last_token_activation)

    return torch.cat(activations, dim=0)


def main(base_model, checkpoint, eval_file, persona_vector, layer, output_file):
    print(f"Calculating finetuning shift...")
    print(f"  Base: {base_model}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Layer: {layer}")

    # Load persona vector
    persona_vec = torch.load(persona_vector, weights_only=False)[layer]

    # Load evaluation data from CSV
    eval_df = pd.read_csv(eval_file)

    # Extract prompts directly from CSV (already formatted)
    prompts = eval_df["prompt"].tolist()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Extract base model activations
    print(f"Loading base model...")
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    base_acts = extract_last_prompt_activations(base_model_obj, tokenizer, prompts, layer)
    base_avg = base_acts.mean(dim=0)
    del base_model_obj
    torch.cuda.empty_cache()

    # Extract checkpoint activations
    print(f"Loading checkpoint...")
    checkpoint_model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    checkpoint_acts = extract_last_prompt_activations(checkpoint_model, tokenizer, prompts, layer)
    checkpoint_avg = checkpoint_acts.mean(dim=0)
    del checkpoint_model
    torch.cuda.empty_cache()

    # Calculate shift
    shift_vector = checkpoint_avg - base_avg

    # Project onto persona
    finetuning_shift = a_proj_b(shift_vector.unsqueeze(0), persona_vec.unsqueeze(0)).item()

    # Additional metrics
    shift_mag = shift_vector.norm().item()
    cos_sim = ((shift_vector * persona_vec).sum() / (shift_vector.norm() * persona_vec.norm())).item()

    results = {
        "finetuning_shift": finetuning_shift,
        "shift_magnitude": shift_mag,
        "cosine_similarity": cos_sim,
        "num_prompts": len(prompts),
        "layer": layer
    }

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Finetuning shift: {finetuning_shift:.4f}")
    print(f"  Saved to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--persona_vector", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    main(
        args.base_model,
        args.checkpoint,
        args.eval_file,
        args.persona_vector,
        args.layer,
        args.output
    )
