import os
import glob
import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def cos_sim(a, b):
    """Compute cosine similarity between two vectors."""
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def plot_vector_similarity(
    trait,
    base_dir="output/persona_vectors",
    layer=None,
    output_dir=None,
    figsize=(10, 8)
):
    """
    Plot a correlation matrix showing cosine similarity between persona vectors
    for a given trait across different variants/models.
    
    Args:
        trait: Trait name (e.g., "anxious")
        base_dir: Base directory containing persona vectors
        layer: Specific layer index to use (None = average across all layers)
        output_dir: Directory to save the figure (None = show plot)
        figsize: Figure size tuple
    """
    # Find all files matching the pattern: {trait}_*_response_avg_diff.pt
    # Need to search for both with and without variant numbers
    pattern_with_variant = os.path.join(base_dir, "**", f"{trait}_*_response_avg_diff.pt")
    pattern_base = os.path.join(base_dir, "**", f"{trait}_response_avg_diff.pt")
    
    files_with_variant = glob.glob(pattern_with_variant, recursive=True)
    files_base = glob.glob(pattern_base, recursive=True)
    
    # Combine and remove duplicates
    files = list(set(files_with_variant + files_base))
    
    if len(files) == 0:
        print(f"No files found matching pattern for trait '{trait}' in {base_dir}")
        print(f"Tried patterns:")
        print(f"  - {pattern_with_variant}")
        print(f"  - {pattern_base}")
        return
    
    # Sort files for consistent ordering
    files.sort()
    
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  - {f}")
    
    # Extract variant names from file paths
    # Pattern: {base_dir}/{model_name}/{trait}_{variant}_response_avg_diff.pt
    variants_data = []  # List of (variant_num, variant_label, vector)
    
    for filepath in files:
        # Extract variant name from filename
        filename = os.path.basename(filepath)
        
        # Check for base pattern: {trait}_response_avg_diff.pt
        base_pattern = f"{trait}_response_avg_diff.pt"
        if filename == base_pattern:
            variant_name = "1"  # Base version is variant 1
        else:
            # Check for variant pattern: {trait}_{variant}_response_avg_diff.pt
            variant_pattern = f"{trait}_"
            suffix = "_response_avg_diff.pt"
            
            if filename.startswith(variant_pattern) and filename.endswith(suffix):
                # Extract the variant part between trait_ and _response_avg_diff.pt
                variant_part = filename[len(variant_pattern):-len(suffix)]
                variant_name = variant_part if variant_part else "1"
            else:
                print(f"Warning: Filename '{filename}' doesn't match expected pattern, skipping")
                continue
        
        variant_label = f"{trait}_{variant_name}"
        
        try:
            vector = torch.load(filepath)
            variants_data.append((variant_label, vector))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    if len(variants_data) < 2:
        print(f"Need at least 2 vectors to compute similarity matrix. Found {len(variants_data)}")
        return
    
    # Sort by variant number
    variants_data.sort(key=lambda x: x[0])
    
    # Extract sorted variants and vectors
    variants = [v[0] for v in variants_data]
    vectors = [v[1] for v in variants_data]
    
    # Handle layer selection
    if layer is not None:
        # Use specific layer
        vectors = [v[layer] for v in vectors]
    else:
        # Average across all layers
        vectors = [v.mean(dim=0) for v in vectors]
    
    # Compute pairwise cosine similarity
    n = len(vectors)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = cos_sim(vectors[i], vectors[j])
                if isinstance(sim, torch.Tensor):
                    sim = sim.item()
                similarity_matrix[i, j] = sim
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        similarity_matrix,
        xticklabels=variants,
        yticklabels=variants,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    layer_label = f" (layer {layer})" if layer is not None else " (avg across layers)"
    plt.title(f"Cosine Similarity Matrix: {trait} persona vectors", fontsize=14)
    plt.xlabel("Variant", fontsize=12)
    plt.ylabel("Variant", fontsize=12)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"vector_sim_{trait}.png"), dpi=300, bbox_inches='tight')
        print(f"Figure saved to {os.path.join(output_dir, f"vector_sim_{trait}.png")}")
    else:
        plt.show()
    
    return similarity_matrix, variants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cosine similarity matrix for persona vectors")
    parser.add_argument("--trait", type=str, required=True, help="Trait name (e.g., 'anxious')")
    parser.add_argument("--base-dir", type=str, default="output/persona_vectors/Qwen/Qwen3-4B-Instruct-2507", 
                       help="Base directory containing persona vectors")
    parser.add_argument("--layer", type=int, default=20, 
                       help="Specific layer to use (None = average across all layers)")
    parser.add_argument("--output_dir", type=str, default="figs/", 
                       help="Output path for the figure")
    
    args = parser.parse_args()
    
    plot_vector_similarity(
        trait=args.trait,
        base_dir=args.base_dir,
        layer=args.layer,
        output_dir=args.output_dir
    )
