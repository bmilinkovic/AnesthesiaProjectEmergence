#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from analyze_similarity_zeros import create_composite_figure

# Constants
RESULTS_DIR = "results/similarity_matrices"

def create_composite_figure(similarity_fig_path, zeros_fig_path, save_path):
    """
    Create a composite figure by combining two existing figures.
    
    Parameters
    ----------
    similarity_fig_path : str
        Path to the similarity matrices figure
    zeros_fig_path : str
        Path to the zero proportions figure
    save_path : str
        Path to save the composite figure
    """
    # Create figure with GridSpec
    fig = plt.figure(figsize=(10, 16))  # Adjusted size for better proportions
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)  # Changed ratio to 3:1 and reduced spacing
    
    # Load and plot similarity matrices figure
    ax1 = fig.add_subplot(gs[0])
    img1 = mpimg.imread(similarity_fig_path)
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.text(-0.02, 1.02, 'a.', transform=ax1.transAxes, fontsize=16, fontweight='bold')
    
    # Load and plot zero proportions figure
    ax2 = fig.add_subplot(gs[1])
    img2 = mpimg.imread(zeros_fig_path)
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.text(-0.02, 1.02, 'b.', transform=ax2.transAxes, fontsize=16, fontweight='bold')
    
    # Adjust layout to minimize spacing
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.05)
    
    # Save as PNG and SVG
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + '.svg', bbox_inches='tight')
    plt.close()

def main():
    # Process each condition
    conditions = ['Propofol', 'Xenon', 'Ketamine']
    
    for condition in conditions:
        print(f"Creating composite figure for {condition}...")
        
        # Define paths
        similarity_fig = os.path.join(RESULTS_DIR, f'{condition}_similarity_matrices.png')
        zeros_fig = os.path.join(RESULTS_DIR, f'{condition}_zero_proportions.png')
        composite_path = os.path.join(RESULTS_DIR, f'composite_{condition.lower()}')
        
        # Create composite figure
        create_composite_figure(similarity_fig, zeros_fig, composite_path)
        print(f"Saved composite figure as {composite_path}.png/svg")

if __name__ == "__main__":
    main() 