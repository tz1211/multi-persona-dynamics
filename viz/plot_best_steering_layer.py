import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_steering_effect_by_layer(
    base_dir="output/eval_persona_eval/Qwen/Qwen3-4B-Instruct-2507",
    traits=["critical", "pessimistic", "sycophantic"],
    output_path="figs/steering_layer_plots.png",
    plot_coherence=False
):
    """
    Plot steering effect for each trait across different layers and coefficients.
    
    Args:
        base_dir: Base directory containing the CSV files
        traits: List of traits to plot
        output_path: Path to save the output figure
        plot_coherence: If True, plot coherence instead of trait expression score
    """
    # Find all steer_response files
    pattern = os.path.join(base_dir, "*_steer_response_layer_*_coef_*.csv")
    all_files = glob.glob(pattern)
    
    # Filter for specified traits
    trait_files = {}
    for trait in traits:
        trait_files[trait] = [f for f in all_files if f"/{trait}_steer_response" in f]
    
    # Parse files and extract data: {trait: {coef: {layer: mean_score}}}
    data = {trait: {} for trait in traits}
    
    for trait in traits:
        for filepath in trait_files[trait]:
            # Parse layer and coefficient from filename: {trait}_steer_response_layer_{layer}_coef_{coef}.csv
            match = re.search(rf"{trait}_steer_response_layer_(\d+)_coef_([\d.]+)\.csv", filepath)
            if not match:
                continue
            
            layer = int(match.group(1))
            coef = float(match.group(2))
            
            try:
                df = pd.read_csv(filepath)
                if plot_coherence:
                    if 'coherence' not in df.columns:
                        print(f"Warning: 'coherence' column not found in {filepath}")
                        continue
                    mean_score = df['coherence'].mean()
                else:
                    if trait not in df.columns:
                        print(f"Warning: Trait '{trait}' not found in {filepath}")
                        continue
                    mean_score = df[trait].mean()
                
                if coef not in data[trait]:
                    data[trait][coef] = {}
                data[trait][coef][layer] = mean_score
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes = axes.flatten()
    
    # Get all unique coefficients and define discrete colors
    coefficients = sorted(set(coef for trait_data in data.values() for coef in trait_data.keys()))
    
    # Discrete colors for each coefficient value (dark red to orange-yellow)
    discrete_colors = {
        0.5: '#5A0A15',   # Very dark red
        1.0: '#7A1A4A',   # Dark purple-red
        1.5: '#8B2D8B',   # Vibrant purple
        2.0: '#E64A19',   # Bright orange-red
        2.5: '#FFA500',   # Orange-yellow
    }
    
    # Map coefficients to colors, with fallback for unexpected values
    coef_to_color = {}
    for coef in coefficients:
        if coef in discrete_colors:
            coef_to_color[coef] = discrete_colors[coef]
        else:
            # Fallback: use plasma colormap for unexpected coefficient values
            norm_coef = (coef - min(coefficients)) / (max(coefficients) - min(coefficients)) if len(coefficients) > 1 else 0.5
            coef_to_color[coef] = plt.cm.plasma(0.2 + 0.7 * norm_coef)
    
    # Collect all data to determine shared axis limits across all plots
    all_x_values = []
    all_y_values = []
    all_x_ticks = set()
    
    for trait in traits:
        all_layers = sorted(set(layer for coef_data in data[trait].values() for layer in coef_data.keys()))
        
        if all_layers:
            all_x_values.extend(all_layers)
            min_layer, max_layer = all_layers[0], all_layers[-1]
            
            # Create ticks every 5 layers
            start_tick = (min_layer // 5) * 5
            end_tick = ((max_layer + 4) // 5) * 5
            all_x_ticks.update(range(start_tick, end_tick + 1, 5))
            
            # Collect all y-values for this trait
            for coef in sorted(coefficients):
                if coef in data[trait]:
                    for layer in all_layers:
                        if layer in data[trait][coef]:
                            all_y_values.append(data[trait][coef][layer])
    
    # Calculate shared axis limits with 5% padding
    shared_xlim = None
    shared_ylim = None
    if all_x_values:
        x_min, x_max = min(all_x_values), max(all_x_values)
        x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 1
        shared_xlim = (x_min - x_padding, x_max + x_padding)
    
    if all_y_values:
        y_min, y_max = min(all_y_values), max(all_y_values)
        y_padding = (y_max - y_min) * 0.05 if y_max > y_min else 1
        shared_ylim = (y_min - y_padding, y_max + y_padding)
    
    shared_x_ticks = sorted(all_x_ticks) if all_x_ticks else None
    
    # Plot each trait
    for idx, trait in enumerate(traits):
        ax = axes[idx]
        
        all_layers = sorted(set(layer for coef_data in data[trait].values() for layer in coef_data.keys()))
        
        if not all_layers:
            ax.text(0.5, 0.5, f'No data for {trait}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(trait.capitalize(), fontsize=14, fontweight='bold')
            if shared_xlim:
                ax.set_xlim(shared_xlim)
            if shared_ylim:
                ax.set_ylim(shared_ylim)
            continue
        
        # Plot lines for each coefficient
        for coef in sorted(coefficients):
            if coef not in data[trait]:
                continue
            
            layers = []
            scores = []
            for layer in all_layers:
                if layer in data[trait][coef]:
                    layers.append(layer)
                    scores.append(data[trait][coef][layer])
            
            if layers:
                ax.plot(layers, scores, 
                       marker='o', 
                       label=f'{coef:.1f}',
                       color=coef_to_color[coef],
                       linewidth=2,
                       markersize=6)
        
        ax.set_xlabel('Layer', fontsize=12)
        ylabel = 'Coherence' if plot_coherence else 'Trait expression score'
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(trait.capitalize(), fontsize=14, fontweight='bold')
        
        # Apply shared axis limits for consistent plot sizes
        if shared_xlim:
            ax.set_xlim(shared_xlim)
        if shared_ylim:
            ax.set_ylim(shared_ylim)
        
        # Set x-axis ticks
        if shared_x_ticks:
            ax.set_xticks(shared_x_ticks)
            ax.set_xticklabels([t for t in shared_x_ticks])
        elif all_layers:
            ax.set_xticks(all_layers)
            ax.set_xticklabels([l for l in all_layers])
        
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Layer-wise steering with persona vectors', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    # Add discrete colorbar on the left, centered between the two rows
    if len(coefficients) > 1:
        # Get positions after tight_layout
        pos_top = axes[0].get_position()  # top-left subplot
        pos_bottom = axes[1].get_position()  # bottom-left subplot
        
        # Calculate center position between top and bottom rows
        center_y = (pos_top.y0 + pos_bottom.y1) / 2
        height = pos_top.y1 - pos_bottom.y0 
        
        # Position colorbar to the left of the leftmost subplots
        cax = fig.add_axes([-0.05, center_y - height/4, 0.015, height * 0.5])
        
        # Create discrete colormap from coefficient-color mapping
        color_list = [coef_to_color[coef] for coef in coefficients]
        cmap = ListedColormap(color_list)
        
        # Create boundaries for discrete colorbar (one boundary between each coefficient)
        # Ensure first and last ticks are centered by extending boundaries symmetrically
        boundaries = []
        for i in range(len(coefficients) - 1):
            boundaries.append((coefficients[i] + coefficients[i+1]) / 2)
        
        # Extend boundaries so first and last coefficients are centered
        if len(coefficients) > 1:
            first_interval = boundaries[0] - coefficients[0]
            last_interval = coefficients[-1] - boundaries[-1]
            boundaries = [coefficients[0] - first_interval] + boundaries + [coefficients[-1] + last_interval]
        else:
            boundaries = [coefficients[0] - 0.1, coefficients[0] + 0.1]
        
        norm = BoundaryNorm(boundaries, cmap.N)
        
        cbar = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical', 
                           boundaries=boundaries, ticks=coefficients)
        cbar.set_label('Steering coefficient', fontsize=11, rotation=90, labelpad=-65)
        cbar.set_ticklabels([f'{c:.2f}' for c in coefficients])
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    return fig, axes


if __name__ == "__main__":
    # Plot trait expression scores
    plot_steering_effect_by_layer()
    
    # Plot coherence
    base_output = "figs/steering_layer_plots.png"
    coherence_output = base_output.replace(".png", "_coherence.png")
    plot_steering_effect_by_layer(output_path=coherence_output, plot_coherence=True)
