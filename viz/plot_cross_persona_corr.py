import os
import re
import glob
import torch
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


def cos_sim(a, b):
    """Compute cosine similarity between two vectors."""
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))

def get_vector(persona: str, layer: int = 20):
    """Get the vector for a given persona."""
    vectors = torch.load(f"output/persona_vectors/Qwen/Qwen3-4B-Instruct-2507/{persona}_response_avg_diff.pt")
    return vectors[layer]

def get_last_checkpoint(checkpoint_dir: str, persona: str):
    # Normalize the directory path (remove trailing slash if present)
    checkpoint_dir = os.path.normpath(checkpoint_dir)
    
    # Find all checkpoints in the directory
    pattern = os.path.join(checkpoint_dir, f"checkpoint-*_{persona}.csv")
    checkpoint_paths = glob.glob(pattern)
    
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found matching '{pattern}' in {checkpoint_dir}")

    # Get the last checkpoint by extracting numbers and sorting numerically
    def _get_checkpoint_number(path):
        """Extract checkpoint number from path like 'checkpoint-50_optimistic.csv' -> 50"""
        basename = os.path.basename(path)
        match = re.match(r"checkpoint-(\d+)", basename)
        if match:
            return int(match.group(1))
        return float('inf')  # Invalid checkpoint, will be sorted last
    
    # Sort by checkpoint number and get the last one
    checkpoint_paths.sort(key=_get_checkpoint_number)
    return checkpoint_paths[-1]


def get_delta_trait_score_and_ft_shift(persona: str, trait: str): 
    # Get baseline llm-judge score for trait 
    df_baseline = pd.read_csv(f"output/eval_persona_eval/Qwen/Qwen3-4B-Instruct-2507/{trait}_baseline.csv")
    baseline_score = df_baseline[trait].mean()

    # Get finetuned llm-judge score for trait 
    finetuned_dir = f"output/{persona}_projection_eval" 
    last_checkpoint_path = get_last_checkpoint(finetuned_dir, trait)
    df_finetuned = pd.read_csv(last_checkpoint_path)
    finetuned_score = df_finetuned[trait].mean()
    # Get finetuning shift for trait
    finetuning_shift = df_finetuned[f"{last_checkpoint_path.split('/')[-1].split('.')[0]}_response_avg_diff_prompt_last_proj_layer20_finetuning_shift"].mean()
    
    # Get delta llm-judge score for trait
    return finetuned_score - baseline_score, finetuning_shift


def get_one_datapoint(persona_a: str, persona_b: str): 
    """
    Given persona_a that we're finetuning on, and persona_b that we're evaluating,
    return: 
    - the relative change in llm-judge score for trait expressiveness 
    - the relative change in finetuning shift
    - the cosine similarity score of the corresponding vectors
    """
    # Get normalised delta trait score and ft shift
    delta_trait_score_a, _ = get_delta_trait_score_and_ft_shift(persona_a, persona_a)
    delta_trait_score_b, ft_shift_b = get_delta_trait_score_and_ft_shift(persona_a, persona_b)
    normalised_delta_trait_score = delta_trait_score_b / delta_trait_score_a
    
    # Get cosine similarity score of the corresponding vectors
    vector_a = get_vector(persona_a, layer=20)
    vector_b = get_vector(persona_b, layer=20)
    cosine_similarity = cos_sim(vector_a, vector_b)
    
    return normalised_delta_trait_score, ft_shift_b, cosine_similarity


def main(persona_a_list: list[str], persona_b_list: list[str]):
    """
    Main function to plot the correlation between the normalised delta trait score and the cosine similarity score.
    """
    delta_trait_score_list = []
    ft_shift_list = []
    cosine_similarity_list = []
    for persona_a, persona_b in zip(persona_a_list, persona_b_list):
        try:
            normalised_delta_trait_score, ft_shift_b, cosine_similarity = get_one_datapoint(persona_a, persona_b)
            delta_trait_score_list.append(normalised_delta_trait_score)
            ft_shift_list.append(ft_shift_b)
            cosine_similarity_list.append(cosine_similarity)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Skipping {persona_a} -> {persona_b}: {e}")
            continue

    # Plot and save the correlation between the normalised delta trait score and the cosine similarity score
    plt.figure(figsize=(8, 6))
    plt.scatter(cosine_similarity_list, delta_trait_score_list, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # Add line of best fit
    if len(cosine_similarity_list) > 1:
        z = np.polyfit(cosine_similarity_list, delta_trait_score_list, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(cosine_similarity_list), max(cosine_similarity_list), 100)
        plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')
    
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Normalised Delta Trait Score", fontsize=12)
    plt.title("Correlation between Cosine Similarity and Normalised Delta Trait Score", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figs/cos_sim_vs_normalised_delta_trait_score.png", dpi=300)
    plt.close()
    print("Plot saved to figs/cos_sim_vs_normalised_delta_trait_score.png")

    # Plot and save the correlation between the ft shift and the cosine similarity score
    plt.figure(figsize=(8, 6))
    plt.scatter(cosine_similarity_list, ft_shift_list, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # Add line of best fit
    if len(cosine_similarity_list) > 1:
        z = np.polyfit(cosine_similarity_list, ft_shift_list, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(cosine_similarity_list), max(cosine_similarity_list), 100)
        plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')
    
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Finetuning Shift", fontsize=12)
    plt.title("Correlation between Cosine Similarity and Finetuning Shift", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figs/cos_sim_vs_finetuning_shift.png", dpi=300)
    plt.close()
    print("Plot saved to figs/cos_sim_vs_finetuning_shift.png")
    return delta_trait_score_list, ft_shift_list, cosine_similarity_list


if __name__ == "__main__":
    persona_list = [
        "confident",
        "critical", 
        "optimistic", 
        "pessimistic", 
        "sycophantic"
    ]
    persona_a_list = []
    persona_b_list = []
    for persona_a in persona_list:
        for persona_b in persona_list:
            if persona_a != persona_b:
                persona_a_list.append(persona_a)
                persona_b_list.append(persona_b)
    
    main(persona_a_list, persona_b_list)