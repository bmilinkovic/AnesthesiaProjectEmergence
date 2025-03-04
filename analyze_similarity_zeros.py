#!/usr/bin/env python3
"""
Script for analyzing the proportion of zeros in similarity matrices across conditions
and performing statistical comparisons between Wake and each condition.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import wilcoxon, chi2, norm
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import combine_pvalues
import matplotlib.image as mpimg

# Constants
DATA_DIR = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/ssdiData/preopt"
RESULTS_DIR = "results/similarity_matrices"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_similarity_matrices(condition_dir):
    """
    Load similarity matrices for Wake and condition from the specified directory.
    
    Parameters
    ----------
    condition_dir : str
        Path to the condition directory containing both Wake and condition files
        
    Returns
    -------
    wake_matrices : dict
        Dictionary of Wake matrices for each participant
    condition_matrices : dict
        Dictionary of condition matrices for each participant
    participant_ids : list
        List of participant IDs
    """
    # Find all preopt_optima_dist files
    files = [f for f in os.listdir(condition_dir) if "preopt_optima_dist" in f and f.endswith(".mat")]
    print(f"Found {len(files)} preopt_optima_dist files in {condition_dir}")
    print("Sample files:")
    for f in files[:3]:
        print(f"  {f}")
    
    # Group files by participant
    participants = {}
    for f in files:
        # Extract participant ID (e.g., H0010 from H0010P_mdim_2_preopt_optima_dist.mat)
        participant_id = f[:5]
        if participant_id not in participants:
            participants[participant_id] = {'W': [], 'C': []}
        
        # Determine if this is a Wake (W) or condition (C) file
        if 'W_mdim' in f:
            participants[participant_id]['W'].append(f)
        else:
            participants[participant_id]['C'].append(f)
    
    # Find valid participants (those with both Wake and condition files)
    valid_participants = [p for p in participants if participants[p]['W'] and participants[p]['C']]
    print(f"Found {len(valid_participants)} valid participants")
    print(f"Participant IDs: {valid_participants}")
    
    # Load matrices for valid participants
    wake_matrices = {}
    condition_matrices = {}
    
    for participant_id in valid_participants:
        wake_files = participants[participant_id]['W']
        condition_files = participants[participant_id]['C']
        
        # Load Wake matrices
        wake_matrices[participant_id] = {}
        for f in wake_files:
            macro = int(f.split('_mdim_')[1].split('_')[0])
            mat_data = loadmat(os.path.join(condition_dir, f))
            wake_matrices[participant_id][macro] = mat_data['goptp']
        
        # Load condition matrices
        condition_matrices[participant_id] = {}
        for f in condition_files:
            macro = int(f.split('_mdim_')[1].split('_')[0])
            mat_data = loadmat(os.path.join(condition_dir, f))
            condition_matrices[participant_id][macro] = mat_data['goptp']
    
    return wake_matrices, condition_matrices, valid_participants

def calculate_zero_proportions(matrices):
    """
    Calculate the proportion of zeros and near-zeros in similarity matrices for each participant.
    
    Parameters:
    -----------
    matrices : dict
        Dictionary containing matrices for each participant and macro scale
    
    Returns:
    --------
    dict
        Dictionary with participant IDs as keys and zero proportions as values
    """
    ZERO_THRESHOLD = 1e-6  # Values below this are considered effectively zero
    zero_proportions = {}
    
    for participant, macro_matrices in matrices.items():
        total_elements = 0
        total_zeros = 0
        
        # Print debug info for the first participant
        if len(zero_proportions) == 0:
            print(f"\nDebug info for first participant {participant}:")
            for macro, matrix in macro_matrices.items():
                n = matrix.shape[0]
                exact_zeros = np.sum(matrix == 0)
                near_zeros = np.sum((matrix > 0) & (matrix < ZERO_THRESHOLD))
                
                print(f"Macro {macro}:")
                print(f"  Shape: {matrix.shape}")
                print(f"  Exact zeros: {exact_zeros}")
                print(f"  Near zeros: {near_zeros}")
                print(f"  Total elements: {n * n}")
                print(f"  Proportion zeros: {(exact_zeros + near_zeros) / (n * n):.6f}")
                
                # Print value distribution
                percentiles = np.percentile(matrix.flatten(), [0, 25, 50, 75, 100])
                print(f"  Value distribution:")
                print(f"    Min (0th): {percentiles[0]:.6f}")
                print(f"    25th: {percentiles[1]:.6f}")
                print(f"    Median: {percentiles[2]:.6f}")
                print(f"    75th: {percentiles[3]:.6f}")
                print(f"    Max (100th): {percentiles[4]:.6f}")
                
                if macro == 2:  # Print a small sample of the matrix
                    print("  Sample values:")
                    print(matrix[:5, :5])
                print()
        
        # Calculate proportion of zeros and near-zeros for all matrices
        for matrix in macro_matrices.values():
            n = matrix.shape[0]
            total_elements += n * n
            total_zeros += np.sum(matrix == 0) + np.sum((matrix > 0) & (matrix < ZERO_THRESHOLD))
        
        zero_proportions[participant] = total_zeros / total_elements
    
    return zero_proportions

def create_comparison_plot(wake_props, condition_props, condition_name, ax=None):
    """
    Create a plot comparing zero proportions between Wake and condition.
    
    Parameters:
    -----------
    wake_props : dict
        Dictionary of Wake zero proportions by participant
    condition_props : dict
        Dictionary of condition zero proportions by participant
    condition_name : str
        Name of the condition being compared to Wake
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, current axes will be used.
    
    Returns:
    --------
    float
        p-value from statistical test
    """
    # Prepare data for plotting
    participants = sorted(wake_props.keys())
    wake_values = [wake_props[p] for p in participants]
    condition_values = [condition_props[p] for p in participants]
    
    if ax is None:
        ax = plt.gca()
    
    # Plot individual points and lines connecting pairs
    x = np.arange(len(participants))
    ax.plot(x, wake_values, 'o-', label='Wake', color='#00A08A', markersize=12)
    ax.plot(x, condition_values, 'o-', label=condition_name, 
            color={'Propofol': '#F98400', 'Xenon': '#F2AD00', 'Ketamine': '#4B6AA8'}[condition_name],
            markersize=12)
    
    # Connect paired points with gray lines
    for i in range(len(participants)):
        ax.plot([i, i], [wake_values[i], condition_values[i]], 'gray', alpha=0.5)
    
    # Perform statistical tests
    try:
        differences = np.array(wake_values) - np.array(condition_values)
        if np.all(differences == 0):
            p_value = 1.0
        else:
            _, p_value = wilcoxon(wake_values, condition_values, 
                                zero_method='zsplit', alternative='two-sided')
    except Exception as e:
        print(f"Statistical test error: {str(e)}")
        p_value = float('nan')
    
    # Add labels with increased font sizes
    ax.set_xlabel('Participant', fontsize=16)
    ax.set_ylabel('Proportion of Near-zero Values', fontsize=16)
    
    # Customize x-axis with increased font size
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i+1}' for i in range(len(participants))], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # Add legend with larger font
    ax.legend(fontsize=16)
    
    return p_value

def plot_similarity_matrices(matrices_wake, matrices_condition, condition_name, ax):
    """
    Plot similarity matrices for wake and condition states.
    
    Parameters:
    -----------
    matrices_wake : dict
        Dictionary of Wake matrices
    matrices_condition : dict
        Dictionary of condition matrices
    condition_name : str
        Name of the condition being compared to Wake
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    # Get first participant's data for visualization
    participant = sorted(matrices_wake.keys())[0]
    wake_matrix = matrices_wake[participant][2]  # Using macro scale 2
    condition_matrix = matrices_condition[participant][2]
    
    # Create a figure with two side-by-side matrices
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec())
    ax_wake = plt.subplot(gs[0])
    ax_cond = plt.subplot(gs[1])
    
    # Plot matrices
    vmax = max(np.max(wake_matrix), np.max(condition_matrix))
    vmin = min(np.min(wake_matrix), np.min(condition_matrix))
    
    im_wake = ax_wake.imshow(wake_matrix, cmap='viridis', aspect='auto',
                            vmin=vmin, vmax=vmax)
    im_cond = ax_cond.imshow(condition_matrix, cmap='viridis', aspect='auto',
                            vmin=vmin, vmax=vmax)
    
    # Add titles and labels
    ax_wake.set_title('Wake', fontsize=14)
    ax_cond.set_title(condition_name, fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im_cond, ax=[ax_wake, ax_cond])
    cbar.ax.tick_params(labelsize=12)
    
    # Remove axis labels
    ax_wake.set_xticks([])
    ax_wake.set_yticks([])
    ax_cond.set_xticks([])
    ax_cond.set_yticks([])

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
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)  # Reduced hspace
    
    # Load and plot similarity matrices figure
    ax1 = fig.add_subplot(gs[0])
    img1 = mpimg.imread(similarity_fig_path)
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.text(0.0, 1.1, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')  # Adjusted x position
    
    # Load and plot zero proportions figure
    ax2 = fig.add_subplot(gs[1])
    img2 = mpimg.imread(zeros_fig_path)
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.text(0.0, 1.1, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold')  # Aligned with 'a'
    
    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    
    # Save as PNG and SVG
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + '.svg', bbox_inches='tight')
    plt.close()

def combine_pvalues(p_values, method='fisher'):
    """
    Combine multiple p-values into a single test statistic.
    
    Parameters:
    -----------
    p_values : list
        List of p-values to combine
    method : str
        Method to use ('fisher' or 'stouffer')
    
    Returns:
    --------
    tuple
        (combined statistic, combined p-value)
    """
    if method == 'fisher':
        # Fisher's method: -2 * sum(ln(p))
        statistic = -2 * np.sum(np.log(p_values))
        df = 2 * len(p_values)
        p_value = 1 - chi2.cdf(statistic, df)
        return statistic, p_value
    elif method == 'stouffer':
        # Stouffer's Z-score method
        z_scores = norm.ppf(1 - np.array(p_values))
        z_combined = np.sum(z_scores) / np.sqrt(len(z_scores))
        p_value = 1 - norm.cdf(z_combined)
        return z_combined, p_value
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    """Main execution function."""
    # Ensure results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Process each condition
    conditions = ['Propofol', 'Xenon', 'Ketamine']
    condition_dirs = {
        'Propofol': os.path.join(DATA_DIR, 'prop_preopt'),
        'Xenon': os.path.join(DATA_DIR, 'xenon_preopt'),
        'Ketamine': os.path.join(DATA_DIR, 'ket_preopt')
    }
    
    # Store results for summary
    all_results = []
    all_pvalues = []
    
    for condition in conditions:
        print(f"\nProcessing Wake vs. {condition}...")
        
        # Load matrices
        wake_matrices, condition_matrices, participant_ids = load_similarity_matrices(condition_dirs[condition])
        
        if not wake_matrices or not condition_matrices:
            print(f"No valid data found for {condition}")
            continue

        # Calculate zero proportions and perform statistical test
        wake_props = calculate_zero_proportions(wake_matrices)
        condition_props = calculate_zero_proportions(condition_matrices)
        
        # Convert dictionary values to lists while maintaining order
        wake_values = [wake_props[p] for p in participant_ids]
        condition_values = [condition_props[p] for p in participant_ids]
        
        # Calculate statistics
        wake_mean = np.mean(wake_values)
        wake_std = np.std(wake_values)
        condition_mean = np.mean(condition_values)
        condition_std = np.std(condition_values)
        differences = np.array(wake_values) - np.array(condition_values)
        mean_diff = np.mean(differences)
        
        # Perform Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(wake_values, condition_values)
        
        # Store results
        result = {
            'condition': condition,
            'n_participants': len(participant_ids),
            'wake_mean': wake_mean,
            'wake_std': wake_std,
            'condition_mean': condition_mean,
            'condition_std': condition_std,
            'p_value': p_value,
            'differences': differences
        }
        all_results.append(result)
        
        # Store p-value for combined analysis
        if not np.isnan(p_value):
            all_pvalues.append(p_value)
        
        # Print detailed results
        print(f"\nResults for {condition}:")
        print(f"Number of participants: {result['n_participants']}")
        print(f"Wake: Mean = {result['wake_mean']:.3f}, SD = {result['wake_std']:.3f}")
        print(f"Condition: Mean = {result['condition_mean']:.3f}, SD = {result['condition_std']:.3f}")
        print(f"Mean difference (Wake - {condition}): {mean_diff:.3f}")
        print(f"Statistical test p-value: {result['p_value']:.3f}")
        print(f"Individual differences (Wake - Condition):")
        for i, diff in enumerate(differences):
            print(f"  {participant_ids[i]}: {float(diff):.3f}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        create_comparison_plot(wake_props, condition_props, condition, ax)
        
        # Save figure
        save_path = os.path.join(RESULTS_DIR, f'{condition}_zero_proportions')
        plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '.svg', bbox_inches='tight')
        plt.close()
    
    # Compute combined statistics
    if len(all_pvalues) > 0:
        print("\nCombined Analysis:")
        print("------------------")
        
        # Fisher's method
        fisher_stat, fisher_p = combine_pvalues(all_pvalues, method='fisher')
        print(f"Fisher's Combined Test:")
        print(f"  Chi-square statistic: {fisher_stat:.3f}")
        print(f"  Combined p-value: {fisher_p:.3f}")
        
        # Stouffer's method
        stouffer_z, stouffer_p = combine_pvalues(all_pvalues, method='stouffer')
        print(f"\nStouffer's Z-score Method:")
        print(f"  Z-score: {stouffer_z:.3f}")
        print(f"  Combined p-value: {stouffer_p:.3f}")
    
    print("\nScript execution completed!")

if __name__ == "__main__":
    main() 