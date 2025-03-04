#!/usr/bin/env python3
"""
Script for creating a paired dot plot showing the proportion of near-zero values
across all conditions (Wake vs. Propofol, Wake vs. Xenon, Wake vs. Ketamine).
Points from the same participant are connected with lines.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon

# Constants
DATA_DIR = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/ssdiData/preopt"
RESULTS_DIR = "results/similarity_matrices"

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_calculate_zeros(condition_dir):
    """
    Load matrices and calculate proportion of near-zero values for each participant.
    
    Parameters:
    -----------
    condition_dir : str
        Path to the condition directory
        
    Returns:
    --------
    tuple
        (wake_props, condition_props, participant_ids)
        Dictionaries containing proportions for each participant
    """
    ZERO_THRESHOLD = 1e-6
    
    # Find all preopt_optima_dist files
    files = [f for f in os.listdir(condition_dir) if "preopt_optima_dist" in f and f.endswith(".mat")]
    
    # Group files by participant
    participants = {}
    for f in files:
        participant_id = f[:5]
        if participant_id not in participants:
            participants[participant_id] = {'W': [], 'C': []}
        
        if 'W_mdim' in f:
            participants[participant_id]['W'].append(f)
        else:
            participants[participant_id]['C'].append(f)
    
    # Find valid participants
    valid_participants = [p for p in participants if participants[p]['W'] and participants[p]['C']]
    
    # Calculate proportions
    wake_props = {}
    condition_props = {}
    
    for participant_id in valid_participants:
        # Process Wake files
        total_elements_wake = 0
        total_zeros_wake = 0
        for f in participants[participant_id]['W']:
            matrix = loadmat(os.path.join(condition_dir, f))['goptp']
            n = matrix.shape[0]
            total_elements_wake += n * n
            total_zeros_wake += np.sum(matrix == 0) + np.sum((matrix > 0) & (matrix < ZERO_THRESHOLD))
        wake_props[participant_id] = total_zeros_wake / total_elements_wake
        
        # Process condition files
        total_elements_cond = 0
        total_zeros_cond = 0
        for f in participants[participant_id]['C']:
            matrix = loadmat(os.path.join(condition_dir, f))['goptp']
            n = matrix.shape[0]
            total_elements_cond += n * n
            total_zeros_cond += np.sum(matrix == 0) + np.sum((matrix > 0) & (matrix < ZERO_THRESHOLD))
        condition_props[participant_id] = total_zeros_cond / total_elements_cond
    
    return wake_props, condition_props, valid_participants

def create_paired_dot_plot(data, save_path=None):
    """
    Create a paired dot plot showing all conditions vs Wake.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing data for each condition
    save_path : str, optional
        Path to save the figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set background color
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Color scheme
    colors = {
        'Propofol': '#F97000',  # Bright coral/orange
        'Xenon': '#F2AD00',     # Gold
        'Ketamine': '#4B6AA8'   # Medium blue
    }
    
    # Position for each condition pair
    positions = {
        'Propofol': 1,
        'Xenon': 2,
        'Ketamine': 3
    }
    
    # Store statistical results and find global y_max
    stats_results = {}
    global_y_max = 0
    
    # First pass to find global y_max
    for condition in ['Propofol', 'Xenon', 'Ketamine']:
        wake_props = data[condition]['wake']
        cond_props = data[condition]['condition']
        wake_values = [wake_props[p] for p in sorted(wake_props.keys())]
        cond_values = [cond_props[p] for p in sorted(cond_props.keys())]
        global_y_max = max(global_y_max, max(max(wake_values), max(cond_values)))
    
    # Plot each condition
    for condition in ['Propofol', 'Xenon', 'Ketamine']:
        wake_props = data[condition]['wake']
        cond_props = data[condition]['condition']
        participants = sorted(wake_props.keys())
        
        # Base positions
        x_wake = positions[condition] - 0.2
        x_cond = positions[condition] + 0.2
        
        # Plot points and connecting lines
        for participant in participants:
            # Plot connecting line
            ax.plot([x_wake, x_cond], 
                   [wake_props[participant], cond_props[participant]], 
                   color='gray', alpha=0.5, zorder=1)
            
            # Plot Wake point
            ax.scatter(x_wake, wake_props[participant], 
                      color='#00A08A', s=100, zorder=2)
            
            # Plot condition point
            ax.scatter(x_cond, cond_props[participant], 
                      color=colors[condition], s=100, zorder=2)
        
        # Calculate statistics
        wake_values = [wake_props[p] for p in participants]
        cond_values = [cond_props[p] for p in participants]
        try:
            stat, p_value = wilcoxon(wake_values, cond_values)
            stats_results[condition] = p_value
            
            # Add statistical annotation at fixed height
            y_text = global_y_max + 0.05
            if p_value < 0.001:
                p_text = 'p < 0.001'
            elif p_value < 0.01:
                p_text = 'p < 0.01'
            elif p_value < 0.05:
                p_text = 'p < 0.05'
            else:
                p_text = f'p = {p_value:.3f}'
            
            ax.text(positions[condition], y_text, p_text,
                   ha='center', va='bottom', fontsize=16)
        except Exception as e:
            print(f"Statistical test error for {condition}: {str(e)}")
    
    # Customize plot
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Propofol', 'Xenon', 'Ketamine'],
                       fontsize=16, fontweight='bold')  # Bold, simplified labels
    ax.tick_params(axis='y', labelsize=16)  # Increased y-tick font size
    
    # Add labels with increased size and padding
    ax.set_ylabel('Proportion of Near-zero Values', 
                 fontsize=18, fontweight='bold', labelpad=15)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Set fixed y-axis limits
    ax.set_ylim(0, 0.30)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Save as SVG
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    plt.close()

def main():
    """Main execution function."""
    # Ensure results directory exists
    ensure_directory_exists(RESULTS_DIR)
    
    # Process each condition
    conditions = {
        'Propofol': 'prop_preopt',
        'Xenon': 'xenon_preopt',
        'Ketamine': 'ket_preopt'
    }
    
    # Collect data for all conditions
    data = {}
    for condition, dir_name in conditions.items():
        condition_dir = os.path.join(DATA_DIR, dir_name)
        wake_props, condition_props, participants = load_and_calculate_zeros(condition_dir)
        data[condition] = {
            'wake': wake_props,
            'condition': condition_props,
            'participants': participants
        }
    
    # Create and save plot
    save_path = os.path.join(RESULTS_DIR, 'paired_zero_proportions.png')
    create_paired_dot_plot(data, save_path)
    print("Created paired dot plot")

if __name__ == "__main__":
    main() 