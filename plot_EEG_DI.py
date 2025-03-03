#!/usr/bin/env python3
"""
Script for plotting EEG Dynamical Independence (DI) measures across different conditions.
This script loads .mat files containing DI measures and creates box plots with overlaid data points
for visualization of the distributions across different conditions and macro values.
"""

import os
import scipy.io
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind
import itertools

# Constants
DATA_DIR = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/ssdiData"
RESULTS_DIR = "/Users/borjan/code/python/AnesthesiaProjectEmergence/results/ddvalues/"

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_dynamical_dependence_data(data_directory, condition_pair):
    """
    Load dynamical dependence data from .mat files for a specific condition pair.
    
    Parameters:
    -----------
    data_directory : str
        Path to directory containing the .mat files
    condition_pair : tuple
        Tuple of conditions to load (e.g., ('W', 'X') for Wake vs. Xenon)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the loaded data
    """
    # List all files that end with "dynamical_dependence.mat"
    files = [f for f in os.listdir(data_directory)
             if os.path.isfile(os.path.join(data_directory, f)) 
             and f.endswith("dynamical_dependence.mat")]
    
    data_list = []
    
    # Debug information
    print(f"Found {len(files)} files ending with dynamical_dependence.mat")
    
    for macro_value in map(str, range(2, 10)):
        # Get prefixes for the first condition to establish valid participants
        prefixes = set(file[:5] for file in files 
                      if file[5] == condition_pair[1] and file[12] == macro_value)
        
        if prefixes:
            print(f"Found {len(prefixes)} participants for macro {macro_value} and condition {condition_pair[1]}")
        
        # Process both conditions
        for condition in condition_pair:
            matching_files = [file for file in files 
                            if file[5] == condition 
                            and file[12] == macro_value 
                            and file[:5] in prefixes]
            
            for file in matching_files:
                try:
                    full_path = os.path.join(data_directory, file)
                    print(f"Loading {file}")
                    data = loadmat(full_path)['dopto'].squeeze()
                    
                    for value in data:
                        data_list.append({
                            "DD": value,
                            "Participant": file[:5],
                            "Condition": condition,
                            "Macro": int(macro_value)
                        })
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
    
    df = pd.DataFrame(data_list)
    if len(df) == 0:
        print("Warning: No data was loaded!")
    else:
        print(f"Successfully loaded {len(df)} data points")
        print("\nSample of loaded data:")
        print(df.head())
        print("\nData summary:")
        print(df.groupby(['Condition', 'Macro']).agg({'DD': ['count', 'mean', 'std']}))
    
    return df

def create_plot(df, condition_pair, save_path=None, ax=None, panel_label=None, show_xlabel=True):
    """
    Create a box plot with overlaid data points.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    condition_pair : tuple
        Tuple of conditions being compared (e.g., ('W', 'X'))
    save_path : str, optional
        Path to save the figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    panel_label : str, optional
        Label for the panel (e.g., 'a', 'b', 'c')
    show_xlabel : bool, optional
        Whether to show x-axis label (default: True)
    """
    if len(df) == 0:
        print("Error: No data to plot!")
        return
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Define colors and condition names mapping
    condition_names = {
        'W': 'Wake',
        'X': 'Xenon',
        'P': 'Propofol',
        'K': 'Ketamine'
    }
    
    # Define professional color scheme for conditions
    condition_colors = {
        'Wake': '#2ecc71',     # Soft green
        'Xenon': '#95a5a6',    # Muted grey
        'Propofol': '#3498db', # Soft blue
        'Ketamine': '#e74c3c'  # Soft red
    }
    
    # Map the conditions to their full names for the plot
    df = df.copy()
    df['Condition'] = df['Condition'].map(condition_names)
    condition_pair_full = [condition_names[c] for c in condition_pair]
    
    # Create new figure if no axes provided
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    # First plot the data points with jitter
    categories = sorted(df['Macro'].unique())
    offsets = {condition_pair_full[0]: -0.2, condition_pair_full[1]: 0.2}
    
    # Plot data points with matching colors
    colors = {cond: condition_colors[cond] for cond in condition_pair_full}
    for condition in condition_pair_full:
        condition_data = df[df['Condition'] == condition]
        for _, row in condition_data.iterrows():
            macro_index = categories.index(row['Macro'])
            jitter = np.random.uniform(-0.1, 0.1)
            adjusted_x = macro_index + offsets[condition] + jitter
            ax.scatter(adjusted_x, row['DD'],
                      color=colors[condition],
                      alpha=0.07,
                      s=50,
                      zorder=3)
    
    # Create box plot with thinner lines
    box = sns.boxplot(x="Macro", y="DD", hue="Condition",
                     data=df, palette=colors,
                     width=0.7,
                     whis=[0, 100],
                     ax=ax,
                     zorder=2,
                     linewidth=0.6)
    
    # Add significance markers
    y_max = df['DD'].max()
    y_range = df['DD'].max() - df['DD'].min()
    y_offset = y_range * 0.05  # 5% of the range for spacing
    
    for i, macro in enumerate(categories):
        # Determine number of asterisks based on condition and macro scale
        if condition_pair[1] == 'P' and macro in [2, 3]:
            asterisks = '**'
        else:
            asterisks = '***'
        
        # Add asterisks above each pair of box plots
        ax.text(i, y_max + y_offset, asterisks,
                horizontalalignment='center',
                fontsize=12)
    
    # Customize the plot
    if show_xlabel:
        ax.set_xlabel('n-Macro Scale', fontsize=16, fontweight='bold', labelpad=15)
    else:
        ax.set_xlabel('')  # Remove only the label, keep the ticks
    
    ax.set_ylabel('Dynamical Dependence (DD)', fontsize=16, fontweight='bold', labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Adjust y-axis limits to accommodate asterisks
    ax.set_ylim(df['DD'].min() - y_offset, y_max + 2*y_offset)
    
    # Add panel label if provided
    if panel_label:
        ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold')
    
    # Adjust legend - move it lower to avoid overlapping with asterisks
    ax.legend(title='Condition', loc='upper left', bbox_to_anchor=(0.0, 0.85))
    
    # Save individual plot if path provided
    if save_path:
        # Save as PNG
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Save as SVG
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        if ax is None:  # Only close if we created a new figure
            plt.close()

def create_composite_figure(data_frames, condition_pairs, save_path):
    """
    Create a composite figure with three vertical panels.
    
    Parameters:
    -----------
    data_frames : list
        List of DataFrames for each condition pair
    condition_pairs : list
        List of condition pairs
    save_path : str
        Path to save the composite figure
    """
    # Create figure with three vertical panels
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    plt.subplots_adjust(hspace=0.3)  # Adjust spacing between panels
    
    # Plot each condition pair in its panel
    for i, (df, condition_pair) in enumerate(zip(data_frames, condition_pairs)):
        # Only show x-label for bottom panel
        show_xlabel = (i == 2)
        create_plot(df, condition_pair, ax=axes[i], 
                   panel_label=f"{chr(97+i)}.", 
                   show_xlabel=show_xlabel)
    
    # Adjust spacing between panels
    plt.tight_layout()
    
    # Save composite figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # Save as SVG
    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()

def main():
    """Main execution function."""
    # Ensure results directory exists
    ensure_directory_exists(RESULTS_DIR)
    
    # Process each condition pair in specific order: Propofol, Xenon, Ketamine
    conditions = [
        ('W', 'P'),  # Wake vs. Propofol
        ('W', 'X'),  # Wake vs. Xenon
        ('W', 'K')   # Wake vs. Ketamine
    ]
    
    condition_names = {
        'X': 'Xenon',
        'P': 'Propofol',
        'K': 'Ketamine'
    }
    
    # Store DataFrames for composite figure
    all_data = []
    
    for wake, condition in conditions:
        print(f"\nProcessing Wake vs. {condition_names[condition]}...")
        # Load data for this condition pair
        df = load_dynamical_dependence_data(DATA_DIR, (wake, condition))
        
        if len(df) > 0:
            all_data.append(df)
            # Create and save individual plot
            save_path = os.path.join(RESULTS_DIR, f'{condition_names[condition]}_DD_plot.png')
            create_plot(df, (wake, condition), save_path)
            print(f"Created plot for Wake vs. {condition_names[condition]}")
        else:
            print(f"Skipping plot for Wake vs. {condition_names[condition]} due to no data")
    
    # Create composite figure if we have all three conditions
    if len(all_data) == 3:
        composite_base = os.path.join(RESULTS_DIR, 'DD_composite_plot')
        # Save in multiple formats
        create_composite_figure(all_data, conditions, f"{composite_base}.png")
        # Save as PDF
        fig, axes = plt.subplots(3, 1, figsize=(10, 18))
        plt.subplots_adjust(hspace=0.3)
        for i, (df, condition_pair) in enumerate(zip(all_data, conditions)):
            show_xlabel = (i == 2)
            create_plot(df, condition_pair, ax=axes[i], 
                       panel_label=f"{chr(97+i)}", 
                       show_xlabel=show_xlabel)
        plt.tight_layout()
        plt.savefig(f"{composite_base}.pdf", format='pdf', bbox_inches='tight')
        plt.close()
        print("\nCreated composite figure in PNG, SVG, and PDF formats")
    
    print("\nScript execution completed!")

if __name__ == "__main__":
    main()