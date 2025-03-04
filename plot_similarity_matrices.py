#!/usr/bin/env python3
"""
Script for creating a composite figure of similarity matrices across conditions and macro scales.
Matrices are arranged in a grid with Wake matrices on the left and anesthetic condition matrices
on the right. Rows represent macro scales and columns represent participants.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sns
from matplotlib.patches import Rectangle

# Constants
DATA_DIR = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/ssdiData"
RESULTS_DIR = "/Users/borjan/code/python/AnesthesiaProjectEmergence/results/similarity_matrices/"

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_similarity_matrices(data_dir, condition_pair):
    """
    Load similarity matrices for a pair of conditions.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the .mat files
    condition_pair : tuple
        Tuple of conditions to compare (e.g., ('W', 'P') for Wake vs. Propofol)
    
    Returns:
    --------
    dict
        Dictionary containing matrices for each condition, participant, and macro scale
    """
    matrices = {cond: {} for cond in condition_pair}
    
    # List all relevant files
    files = [f for f in os.listdir(data_dir)
             if f.endswith("optima_dist.mat")]
    
    print(f"Found {len(files)} optima_dist.mat files")
    
    # First, find all valid participants (those that have data for both conditions)
    valid_participants = set()
    for macro in range(2, 10):
        for file in files:
            if file[5] in condition_pair and file[12] == str(macro):
                prefix = file[:5]
                # Check if this participant has data for both conditions
                has_both = all(
                    any(f.startswith(prefix) and f[5] == cond and f[12] == str(macro)
                        for f in files)
                    for cond in condition_pair
                )
                if has_both:
                    valid_participants.add(prefix)
    
    print(f"Found {len(valid_participants)} valid participants")
    
    # Now load matrices for valid participants
    for participant in sorted(valid_participants):
        for condition in condition_pair:
            matrices[condition][participant] = {}
            for macro in range(2, 10):
                matching_files = [f for f in files
                                if f.startswith(participant)
                                and f[5] == condition
                                and f[12] == str(macro)]
                
                if matching_files:
                    file_path = os.path.join(data_dir, matching_files[0])
                    try:
                        data = loadmat(file_path)
                        if 'gopto' in data:
                            matrix = data['gopto']
                            matrices[condition][participant][macro] = matrix
                        else:
                            print(f"Warning: 'gopto' not found in {file_path}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")
    
    return matrices

def create_similarity_matrix_plot(wake_matrices, condition_matrices, condition_name,
                                save_path=None):
    """
    Create a composite figure showing similarity matrices for Wake vs. Condition.
    
    Parameters:
    -----------
    wake_matrices : dict
        Dictionary of Wake matrices by participant and macro scale
    condition_matrices : dict
        Dictionary of condition matrices by participant and macro scale
    condition_name : str
        Name of the condition being compared to Wake
    save_path : str, optional
        Path to save the figure
    """
    participants = sorted(wake_matrices.keys())
    macro_scales = range(2, 10)
    n_participants = len(participants)
    n_macros = len(macro_scales)
    
    print(f"Creating plot for {len(participants)} participants")
    
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(20, 24))  # Back to original size
    
    # Create GridSpec with space at bottom for colorbar
    gs = plt.GridSpec(n_macros + 1, n_participants * 2 + 1, figure=fig,
                     height_ratios=[1] * n_macros + [0.3],  # Increased space for colorbar
                     hspace=0.05)  # Back to previous spacing
    
    # Color scheme - using the Darjeeling1 palette colors for backgrounds
    colors = {
        'Wake': '#00A08A',    # Deep teal/emerald green
        'Propofol': '#F97000', # Bright coral/orange
        'Xenon': '#F2AD00',    # Changed to a more reddish color to distinguish from Propofol
        'Ketamine': '#4B6AA8'  # Medium blue
    }
    
    # Get the figure coordinates for the matrix areas
    wake_bbox = gs[:-1, :n_participants].get_position(fig)
    condition_bbox = gs[:-1, n_participants+1:].get_position(fig)
    
    # Calculate extensions for the background rectangles
    top_extension = wake_bbox.height * 0.15  # Extension above the matrices to cover labels
    bottom_extension = wake_bbox.height * 0.02  # Very small extension below matrices
    y_shift_up = wake_bbox.height * 0.02  # Shift entire rectangle up slightly
    
    # Create and add background rectangles with condition-specific alpha values
    wake_rect = Rectangle((wake_bbox.x0, wake_bbox.y0 - bottom_extension + y_shift_up), 
                         wake_bbox.width, 
                         wake_bbox.height + top_extension + bottom_extension,
                         facecolor=colors['Wake'],
                         alpha=0.15,
                         zorder=0)
    
    # Use different alpha for Propofol to make it more distinct
    condition_alpha = 0.25 if condition_name == 'Propofol' else 0.15
    condition_rect = Rectangle((condition_bbox.x0, condition_bbox.y0 - bottom_extension + y_shift_up),
                             condition_bbox.width, 
                             condition_bbox.height + top_extension + bottom_extension,
                             facecolor=colors[condition_name],
                             alpha=condition_alpha,
                             zorder=0)
    
    fig.add_artist(wake_rect)
    fig.add_artist(condition_rect)
    
    # Plot matrices
    for i, macro in enumerate(macro_scales):
        for j, participant in enumerate(participants):
            # Wake matrices (left side)
            if macro in wake_matrices[participant]:
                ax = fig.add_subplot(gs[i, j])
                
                # Flip matrix vertically for y-axis mirror
                matrix = np.flipud(wake_matrices[participant][macro])
                im = ax.imshow(matrix, cmap='bone_r', aspect='equal', vmin=0, vmax=1)
                ax.axis('off')
                
                # Add labels for first row and column
                if i == 0:
                    ax.set_title(f'P{j+1}', fontsize=20, pad=15)
                if j == 0:
                    ax.text(-0.2, 0.5, f'{macro}-MACRO',
                           rotation=90, ha='right', va='center',
                           fontsize=18, transform=ax.transAxes)
            
            # Condition matrices (right side)
            if macro in condition_matrices[participant]:
                ax = fig.add_subplot(gs[i, j + n_participants + 1])
                
                # Flip matrix vertically for y-axis mirror
                matrix = np.flipud(condition_matrices[participant][macro])
                im = ax.imshow(matrix, cmap='bone_r', aspect='equal', vmin=0, vmax=1)
                ax.axis('off')
                
                # Add labels for first row
                if i == 0:
                    ax.set_title(f'P{j+1}', fontsize=20, pad=15)
    
    # Calculate proper positions for titles based on the actual matrix positions
    wake_title_x = wake_bbox.x0 + wake_bbox.width/2
    condition_title_x = condition_bbox.x0 + condition_bbox.width/2
    
    # Add condition labels at the top with more space and larger font
    fig.text(wake_title_x, 0.97, 'Wake', ha='center', va='center', fontsize=24, weight='bold')
    fig.text(condition_title_x, 0.97, condition_name, ha='center', va='center', fontsize=24, weight='bold')
    
    # Add horizontal colorbar at the bottom
    cax = fig.add_subplot(gs[-1, n_participants//2:-(n_participants//2)])
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal', label='Degree of Similarity')
    cbar.ax.tick_params(labelsize=20)  # Increased tick label size
    cbar.set_label('Degree of Similarity', size=22, labelpad=15)  # Increased label size and padding
    
    # Adjust layout with more space at the top
    plt.subplots_adjust(top=0.95)
    
    # Save figure if path provided
    if save_path:
        print(f"Saving figure to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Save as SVG
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    plt.close()

def main():
    """Main execution function."""
    # Ensure results directory exists
    ensure_directory_exists(RESULTS_DIR)
    
    # Process each condition pair
    conditions = [
        ('W', 'P', 'Propofol'),
        ('W', 'X', 'Xenon'),
        ('W', 'K', 'Ketamine')
    ]
    
    for wake, condition, condition_name in conditions:
        print(f"\nProcessing Wake vs. {condition_name}...")
        
        # Load matrices
        matrices = load_similarity_matrices(DATA_DIR, (wake, condition))
        
        if matrices[wake] and matrices[condition]:
            # Create and save plot
            save_path = os.path.join(RESULTS_DIR, f'{condition_name}_similarity_matrices.png')
            create_similarity_matrix_plot(matrices[wake], matrices[condition],
                                       condition_name, save_path)
            print(f"Created plot for Wake vs. {condition_name}")
        else:
            print(f"Skipping plot for Wake vs. {condition_name} due to no data")
    
    print("\nScript execution completed!")

if __name__ == "__main__":
    main() 