#!/usr/bin/env python3
"""
Data visualization script for analyzing and plotting EEG data from anesthesia experiments.
This script handles various plotting tasks including GC-matrix, optimization distances,
and node weights visualization.
"""

# Standard library imports
import os
from glob import glob

# Third-party imports
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
import plot

# Constants
SSDIDATA_DIR = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/ssdiData/"
GC_DIR = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/pwcgc_matrix/"
RESULTS_DIR = "results"

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def move_preopt_files():
    """Move preoptimization files to a dedicated directory."""
    destination_dir = os.path.join(SSDIDATA_DIR, "preopt")
    ensure_directory_exists(destination_dir)
    
    file_pattern = "*preopt_dynamical_dependence.mat"
    for file in glob(os.path.join(SSDIDATA_DIR, file_pattern)):
        destination_file = os.path.join(destination_dir, os.path.basename(file))
        if os.path.exists(file):
            os.rename(file, destination_file)

def plot_gc_matrices():
    """Plot Granger causality matrices for all files."""
    save_dir = os.path.join(RESULTS_DIR, "gc_matrix")
    ensure_directory_exists(save_dir)

    # Load label names
    label_names = np.load(os.path.join(os.getcwd(), "labels_new.npy"))
    
    # Process each file
    for file in glob(os.path.join(GC_DIR, "*.mat")):
        try:
            data = sio.loadmat(file)
            if 'edgeWeightsMatrix' in data:
                figure = plot.plot_gc(data['edgeWeightsMatrix'], file, label_names)
                # Save figures handled by plot.plot_gc
            else:
                print(f"Warning: 'edgeWeightsMatrix' not found in {file}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

def plot_preopt_distances():
    """Plot preoptimization and optimization sub-optima distances."""
    for directory in ['results/opt/notitle/png/', 'results/opt/notitle/svg/']:
        ensure_directory_exists(directory)
    
    for file in glob(os.path.join(SSDIDATA_DIR, "*optima_dist.mat")):
        try:
            data = sio.loadmat(file)
            if 'gopto' in data:
                figure = plot.plot_preopt_dist(data['gopto'], file)
                filename_save = f"{os.path.basename(file)[:-4]}_similarity_matrix_plot"
                
                # Save figures
                figure.savefig(os.path.join('results/opt/notitle/svg/', f"{filename_save}.svg"), format='svg')
                figure.savefig(os.path.join('results/opt/notitle/png/', f"{filename_save}.png"), format='png')
            else:
                print(f"Warning: 'gopto' not found in {file}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

def plot_node_weights():
    """Plot node weights and corresponding edge weights."""
    labels = np.load(os.path.join(os.getcwd(), "labels_new.npy"))
    
    for node_file in glob(os.path.join(SSDIDATA_DIR, "*node_weights.mat")):
        try:
            prefix = os.path.basename(node_file)[:6]
            edge_files = glob(os.path.join(GC_DIR, f"pwcgc_matrix_{prefix}*.mat"))
            
            if edge_files:
                # Load node weights
                nweight_data = sio.loadmat(node_file)
                nweight = nweight_data['node_weights']
                
                # Load edge weights
                eweight_data = sio.loadmat(edge_files[0])
                eweight = eweight_data['edgeWeightsMatrix']
                
                # Create plot title
                title = os.path.basename(node_file)[:14].replace("_", " ")
                
                # Generate plot
                plot.plot_nweights(eweight, nweight, title, labels)
            else:
                print(f"No matching edge weights file found for {node_file}")
        except Exception as e:
            print(f"Error processing {node_file}: {str(e)}")

def plot_roi_contributions():
    """Plot ROI contributions to emergent communication subspaces."""
    plt.figure(figsize=(17, 12))
    
    nodedata_dir = os.path.join(os.getcwd(), 'results/data')
    label_names = np.load(os.path.join(os.getcwd(), "labels_new.npy"))
    
    file_patterns = ['2macro', '6macro', '9macro']
    label_order = ['WAKE', 'KET', 'SLEEP', 'XENON', 'PROP']
    
    column_vectors = []
    column_labels = []
    
    for pattern in file_patterns:
        aligned_vectors = []
        aligned_labels = []
        
        for label in label_order:
            matching_files = glob(os.path.join(nodedata_dir, f'*{pattern}*{label}*.mat'))
            
            if matching_files:
                try:
                    data = sio.loadmat(matching_files[0])
                    if 'avg_nodes' in data:
                        aligned_vectors.append(data['avg_nodes'])
                        aligned_labels.append(label)
                    else:
                        print(f"'avg_nodes' not found in {matching_files[0]}")
                except Exception as e:
                    print(f"Error processing {matching_files[0]}: {str(e)}")
        
        column_vectors.extend(aligned_vectors)
        column_labels.extend(aligned_labels)
    
    if column_vectors:
        concatenated_data = np.concatenate(column_vectors, axis=0)
        
        # Create heatmap
        sns.heatmap(concatenated_data.T, cmap='bone_r', yticklabels=label_names)
        
        plt.title("ROI Contribution to Emergent Communication Subspaces",
                 fontsize=18, fontweight='bold', pad=20)
        
        plt.xticks(range(len(column_labels)), column_labels,
                  ha='center', fontsize=12, fontweight='bold')
        plt.gca().set_xticklabels(column_labels, ha='left',
                                 fontsize=12, fontweight='bold')
        
        # Save plots
        ensure_directory_exists('results/heatmaps')
        plt.savefig('results/heatmaps/roi_contribution_all.svg',
                   format='svg', dpi=600)
        plt.savefig('results/heatmaps/roi_contribution_all.png',
                   format='png', dpi=600)
        plt.close()

def main():
    """Main execution function."""
    # Create necessary directories
    ensure_directory_exists(RESULTS_DIR)
    
    # Execute plotting functions
    move_preopt_files()
    plot_gc_matrices()
    plot_preopt_distances()
    plot_node_weights()
    plot_roi_contributions()

if __name__ == "__main__":
    main()
