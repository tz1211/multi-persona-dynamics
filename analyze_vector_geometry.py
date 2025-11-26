"""
Analyze geometric relationships between persona vectors.

This script computes:
1. Cosine similarities between all persona vectors
2. Vector norms and magnitudes
3. Orthogonality tests (similarity ≈ 0)
4. Opposition tests (similarity ≈ -1)
5. Visualizations (heatmaps, PCA, t-SNE)
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return (vec1 @ vec2) / (torch.norm(vec1) * torch.norm(vec2))


def load_persona_vectors(vector_dir, traits, vector_type="response_avg_diff"):
    """Load persona vectors for given traits."""
    vectors = {}
    for trait in traits:
        vector_path = Path(vector_dir) / f"{trait}_{vector_type}.pt"
        if vector_path.exists():
            vectors[trait] = torch.load(vector_path, weights_only=False)
        else:
            print(f"Warning: {vector_path} not found, skipping...")
    return vectors


def analyze_layer(vectors, layer_idx, trait_names):
    """Analyze vectors at a specific layer."""
    # Extract vectors at the specified layer
    layer_vectors = {name: vec[layer_idx].float() for name, vec in vectors.items()}

    # Compute pairwise cosine similarities
    n = len(trait_names)
    similarity_matrix = np.zeros((n, n))

    for i, trait1 in enumerate(trait_names):
        for j, trait2 in enumerate(trait_names):
            if trait1 in layer_vectors and trait2 in layer_vectors:
                sim = cosine_similarity(layer_vectors[trait1], layer_vectors[trait2]).item()
                similarity_matrix[i, j] = sim

    # Compute vector norms
    norms = {name: torch.norm(vec).item() for name, vec in layer_vectors.items()}

    return similarity_matrix, norms, layer_vectors


def analyze_all_layers(vectors, trait_names):
    """Analyze how similarity changes across layers."""
    num_layers = len(next(iter(vectors.values())))

    # For each pair, track similarity across layers
    pair_similarities = {}
    for i, trait1 in enumerate(trait_names):
        for j, trait2 in enumerate(trait_names):
            if i < j and trait1 in vectors and trait2 in vectors:
                pair_name = f"{trait1}_vs_{trait2}"
                similarities = []
                for layer in range(num_layers):
                    vec1 = vectors[trait1][layer].float()
                    vec2 = vectors[trait2][layer].float()
                    sim = cosine_similarity(vec1, vec2).item()
                    similarities.append(sim)
                pair_similarities[pair_name] = similarities

    return pair_similarities


def plot_similarity_heatmap(similarity_matrix, trait_names, layer_idx, output_path):
    """Create heatmap of cosine similarities."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        xticklabels=trait_names,
        yticklabels=trait_names,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(f'Persona Vector Cosine Similarities (Layer {layer_idx})')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def plot_pca(layer_vectors, trait_names, layer_idx, output_path):
    """Create PCA visualization of vectors."""
    # Stack vectors for PCA
    vector_list = [layer_vectors[name].numpy() for name in trait_names if name in layer_vectors]
    vector_matrix = np.stack(vector_list)

    # Apply PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vector_matrix)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=200, alpha=0.6)

    # Add labels
    for i, name in enumerate([n for n in trait_names if n in layer_vectors]):
        plt.annotate(name, (coords[i, 0], coords[i, 1]),
                    fontsize=12, ha='center', va='bottom')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(f'PCA of Persona Vectors (Layer {layer_idx})')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved PCA plot to {output_path}")
    plt.close()


def plot_layer_evolution(pair_similarities, output_path):
    """Plot how similarities evolve across layers."""
    plt.figure(figsize=(12, 6))

    for pair_name, similarities in pair_similarities.items():
        plt.plot(similarities, label=pair_name, marker='o', markersize=3)

    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.title('Persona Vector Similarities Across Layers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Orthogonal')
    plt.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Opposite')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved layer evolution plot to {output_path}")
    plt.close()


def test_hypothesis(similarity_matrix, trait_names, hypothesis_type, pair):
    """Test specific hypotheses about vector relationships."""
    trait1, trait2 = pair
    idx1 = trait_names.index(trait1)
    idx2 = trait_names.index(trait2)
    sim = similarity_matrix[idx1, idx2]

    if hypothesis_type == "orthogonal":
        # Test if similarity is close to 0
        threshold = 0.3
        is_orthogonal = abs(sim) < threshold
        return {
            'pair': f"{trait1} vs {trait2}",
            'similarity': sim,
            'hypothesis': 'orthogonal (independent)',
            'threshold': f'|sim| < {threshold}',
            'result': 'PASS' if is_orthogonal else 'FAIL',
            'interpretation': f"{'Independent' if is_orthogonal else 'Related'} traits"
        }
    elif hypothesis_type == "opposite":
        # Test if similarity is close to -1
        threshold = -0.5
        is_opposite = sim < threshold
        return {
            'pair': f"{trait1} vs {trait2}",
            'similarity': sim,
            'hypothesis': 'opposite (mutually exclusive)',
            'threshold': f'sim < {threshold}',
            'result': 'PASS' if is_opposite else 'FAIL',
            'interpretation': f"{'Opposing' if is_opposite else 'Not strongly opposed'} traits"
        }


def compute_average_similarities(vectors, trait_names):
    """Compute average similarity across all layers for each pair."""
    num_layers = len(next(iter(vectors.values())))
    n = len(trait_names)

    # Accumulate similarities across layers
    avg_similarity_matrix = np.zeros((n, n))

    for layer in range(num_layers):
        similarity_matrix, _, _ = analyze_layer(vectors, layer, trait_names)
        avg_similarity_matrix += similarity_matrix

    # Average across layers
    avg_similarity_matrix /= num_layers

    return avg_similarity_matrix


def main(vector_dir, layer, vector_type="response_avg_diff", output_dir="output/vector_analysis", analyze_all_layers_flag=False):
    """Main analysis function."""
    # Define traits
    traits = ["anxious", "critical", "humorous", "hallucinating", "sycophantic", "funny"]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PERSONA VECTOR GEOMETRIC ANALYSIS")
    print(f"{'='*60}")
    print(f"Vector directory: {vector_dir}")
    print(f"Vector type: {vector_type}")
    print(f"Analyzing layer: {layer}")
    if analyze_all_layers_flag:
        print(f"Also computing averages across ALL layers")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Load vectors
    print("Loading persona vectors...")
    vectors = load_persona_vectors(vector_dir, traits, vector_type)
    available_traits = list(vectors.keys())
    num_layers = len(next(iter(vectors.values())))
    print(f"Loaded vectors for: {', '.join(available_traits)}")
    print(f"Number of layers: {num_layers}\n")

    # Analyze specific layer
    print(f"Analyzing layer {layer}...")
    similarity_matrix, norms, layer_vectors = analyze_layer(vectors, layer, available_traits)

    # Print similarity matrix
    print(f"\n{'='*60}")
    print(f"COSINE SIMILARITY MATRIX (Layer {layer})")
    print(f"{'='*60}")
    df_sim = pd.DataFrame(similarity_matrix, index=available_traits, columns=available_traits)
    print(df_sim.to_string(float_format='%.3f'))

    # Print norms
    print(f"\n{'='*60}")
    print(f"VECTOR NORMS (Layer {layer})")
    print(f"{'='*60}")
    for name, norm in sorted(norms.items()):
        print(f"{name:15s}: {norm:.2f}")

    # Test hypotheses
    print(f"\n{'='*60}")
    print(f"HYPOTHESIS TESTING")
    print(f"{'='*60}\n")

    results = []

    # Test orthogonal trio: anxious, humorous, hallucinating
    if all(t in available_traits for t in ["anxious", "humorous", "hallucinating"]):
        print("Testing ORTHOGONAL TRIO (anxious, humorous, hallucinating):")
        print("-" * 60)
        for pair in [("anxious", "humorous"), ("anxious", "hallucinating"), ("humorous", "hallucinating")]:
            result = test_hypothesis(similarity_matrix, available_traits, "orthogonal", pair)
            results.append(result)
            print(f"{result['pair']:30s} | sim={result['similarity']:6.3f} | {result['result']:4s} | {result['interpretation']}")
        print()

    # Test opposite pair: sycophantic, critical
    if all(t in available_traits for t in ["sycophantic", "critical"]):
        print("Testing OPPOSITE PAIR (sycophantic vs critical):")
        print("-" * 60)
        result = test_hypothesis(similarity_matrix, available_traits, "opposite", ("sycophantic", "critical"))
        results.append(result)
        print(f"{result['pair']:30s} | sim={result['similarity']:6.3f} | {result['result']:4s} | {result['interpretation']}")
        print()

    # Additional interesting pairs
    print("Other notable relationships:")
    print("-" * 60)
    for i, trait1 in enumerate(available_traits):
        for j, trait2 in enumerate(available_traits):
            if i < j:
                sim = similarity_matrix[i, j]
                if abs(sim) > 0.5:  # Moderately strong relationship
                    relationship = "Similar" if sim > 0 else "Opposed"
                    print(f"{trait1:15s} vs {trait2:15s} | sim={sim:6.3f} | {relationship}")

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_path / f"hypothesis_tests_layer{layer}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved hypothesis test results to {results_path}")

    # Create visualizations
    print(f"\n{'='*60}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*60}\n")

    plot_similarity_heatmap(
        similarity_matrix,
        available_traits,
        layer,
        output_path / f"similarity_heatmap_layer{layer}.png"
    )

    plot_pca(
        layer_vectors,
        available_traits,
        layer,
        output_path / f"pca_layer{layer}.png"
    )

    # Analyze across all layers
    print("\nAnalyzing similarity evolution across all layers...")
    pair_similarities = analyze_all_layers(vectors, available_traits)
    plot_layer_evolution(
        pair_similarities,
        output_path / "similarity_across_layers.png"
    )

    # Save layer-wise data
    layer_df = pd.DataFrame(pair_similarities)
    layer_df.to_csv(output_path / "similarities_all_layers.csv", index=False)
    print(f"Saved layer-wise similarities to {output_path / 'similarities_all_layers.csv'}")

    # Compute average similarities across all layers
    if analyze_all_layers_flag:
        print(f"\n{'='*60}")
        print(f"AVERAGE SIMILARITIES ACROSS ALL {num_layers} LAYERS")
        print(f"{'='*60}\n")

        avg_similarity_matrix = compute_average_similarities(vectors, available_traits)

        # Print average similarity matrix
        print("Average Cosine Similarity Matrix:")
        df_avg_sim = pd.DataFrame(avg_similarity_matrix, index=available_traits, columns=available_traits)
        print(df_avg_sim.to_string(float_format='%.3f'))

        # Save average similarities
        df_avg_sim.to_csv(output_path / "average_similarities_all_layers.csv")
        print(f"\nSaved to {output_path / 'average_similarities_all_layers.csv'}")

        # Test hypotheses on average similarities
        print(f"\n{'='*60}")
        print(f"HYPOTHESIS TESTING (AVERAGE ACROSS ALL LAYERS)")
        print(f"{'='*60}\n")

        avg_results = []

        # Test orthogonal trio
        if all(t in available_traits for t in ["anxious", "humorous", "hallucinating"]):
            print("Testing ORTHOGONAL TRIO (anxious, humorous, hallucinating):")
            print("-" * 60)
            for pair in [("anxious", "humorous"), ("anxious", "hallucinating"), ("humorous", "hallucinating")]:
                result = test_hypothesis(avg_similarity_matrix, available_traits, "orthogonal", pair)
                avg_results.append(result)
                print(f"{result['pair']:30s} | sim={result['similarity']:6.3f} | {result['result']:4s} | {result['interpretation']}")
            print()

        # Test opposite pair
        if all(t in available_traits for t in ["sycophantic", "critical"]):
            print("Testing OPPOSITE PAIR (sycophantic vs critical):")
            print("-" * 60)
            result = test_hypothesis(avg_similarity_matrix, available_traits, "opposite", ("sycophantic", "critical"))
            avg_results.append(result)
            print(f"{result['pair']:30s} | sim={result['similarity']:6.3f} | {result['result']:4s} | {result['interpretation']}")
            print()

        # Save average hypothesis results
        avg_results_df = pd.DataFrame(avg_results)
        avg_results_df.to_csv(output_path / "hypothesis_tests_average_all_layers.csv", index=False)
        print(f"Saved average hypothesis results to {output_path / 'hypothesis_tests_average_all_layers.csv'}")

        # Create heatmap for average similarities
        plot_similarity_heatmap(
            avg_similarity_matrix,
            available_traits,
            "average",
            output_path / "similarity_heatmap_average_all_layers.png"
        )

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze geometric relationships between persona vectors")
    parser.add_argument("--vector_dir", type=str, default="output/persona_vectors/Qwen/Qwen3-4B",
                        help="Directory containing persona vectors")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer to analyze in detail (default: 20)")
    parser.add_argument("--vector_type", type=str, default="response_avg_diff",
                        choices=["response_avg_diff", "prompt_avg_diff", "prompt_last_diff"],
                        help="Type of vector to analyze")
    parser.add_argument("--output_dir", type=str, default="output/vector_analysis",
                        help="Output directory for results")
    parser.add_argument("--all_layers", action="store_true",
                        help="Also compute average similarities across all layers")

    args = parser.parse_args()
    main(args.vector_dir, args.layer, args.vector_type, args.output_dir, args.all_layers)
