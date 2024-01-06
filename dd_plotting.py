#%% IMPORTS
import os
import time as tm
from glob import glob

import numpy as np
import pandas as pd
import scipy as sc
import scipy.io as sio

from matplotlib import pyplot as plt
import seaborn as sns

import mne
import plot
import os
import shutil
import os
import glob
import os
import glob
import seaborn as sns
import numpy as np
import seaborn as sns
import seaborn as sns
import numpy as np
import glob

#%% LOADING DATA

# set directories
ssdidata_dir = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/ssdiData/" # <-- SSDI data dir
gc_dir = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/pwcgc_matrix/" # <-- GC data dir


# %% 
# VERY USEFUL PIECE OF CODE!!!!!

# move all files in the director ssdidata_dir that end with "preopt_dynamical_dependence.mat" to a new folder called preopt.

# Define the source directory
source_dir = ssdidata_dir

# Define the destination directory
destination_dir = os.path.join(source_dir, "preopt")

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Get a list of all files in the source directory that end with "preopt_dynamical_dependence.mat"
file_list = [f for f in os.listdir(source_dir) if f.endswith("preopt_dynamical_dependence.mat")]

# Move each file to the destination directory
for file in file_list:
    source_file = os.path.join(source_dir, file)
    destination_file = os.path.join(destination_dir, file)
    shutil.move(source_file, destination_file)


#%% ######################  1. FOR GC-MATRIX PLOTTING!

# Create the directory if it doesn't exist
save_dir = os.path.join(os.getcwd(), "results", "gc_matrix")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Get a list of all files in the directory
file_list = glob(gc_dir + "*.mat")
# Load each file using scipy.io.loadmat and store in a list
data_list = [sc.io.loadmat(file) for file in file_list]


label_names_path = os.path.join(os.getcwd(), "labels_new.npy")
label_names = np.load(label_names_path)

# Loop over each file and plot the data
for i, data in enumerate(data_list):
    if 'edgeWeightsMatrix' in data:
        eweight = data['edgeWeightsMatrix']
    else:
        print("Key 'edgeWeightsMatrix' not found in dictionary.")
    filename = file_list[i]                # <-- get filename from file_list
    figure = plot.plot_gc(eweight, filename, label_names)         # <-- plot the edge weights
    # figure.savefig(os.path.join(os.getcwd(), 'results/gc_matrix/svg/', f"{filename}.svg"), format='svg')
    # figure.savefig(os.path.join(os.getcwd(), 'results/gc_matrix/png/', f"{filename}.png"), format='png')




#%% ####################### 2. PREOPTIMISATION & OPTIMISATION SUB-OPTIMA DISTANCE PLOTTING

preopto_dist_list = glob(ssdidata_dir + "*optima_dist.mat")
preopto_dist_data_list = [sc.io.loadmat(file) for file in preopto_dist_list]

if not os.path.exists('results/opt/notitle/png/'):
    os.makedirs('results/opt/notitle/png/')

if not os.path.exists('results/opt/notitle/svg/'):
    os.makedirs('results/opt/notitle/svg/')

for i, data in enumerate(preopto_dist_data_list):
    if 'gopto' in data:
        preoptdist = data['gopto']
    else:
        print("Key 'gopto' not found in dictionary.")
    filename = preopto_dist_list[i]
    figure = plot.plot_preopt_dist(preoptdist, filename)  # plot the data
    filename_save = f"{filename[108:122]}opt_similarity_matrix_plot"
    figure.savefig(os.path.join(os.getcwd(), 'results/opt/notitle/svg/', f"{filename_save}.svg"), format='svg')
    figure.savefig(os.path.join(os.getcwd(), 'results/opt/notitle/png/', f"{filename_save}.png"), format='png')
# %% #################### 3. n-MACRO PLOTTING

label_names_path = os.path.join(os.getcwd(), "labels_new.npy")
labels = np.load(label_names_path)


# Get a list of all node weights files in the ssdidata_dir directory
node_weights_files = glob.glob(os.path.join(ssdidata_dir, "*node_weights.mat"))

# Load the node weights and edge weights files
for node_weights_file in node_weights_files:
    # Extract the first 6 characters of the node weights file name
    node_weights_prefix = os.path.basename(node_weights_file)[:6]
    # Find the corresponding edge weights file in the gc_dir directory
    edge_weights_file = os.path.join(gc_dir, f"pwcgc_matrix_{node_weights_prefix}*.mat")
    edge_weights_files = glob.glob(edge_weights_file)

    if len(edge_weights_files) > 0:
        # Load the node weights file
        nweight = sio.loadmat(node_weights_file)
        nweight = nweight['node_weights']
        
        # Load the corresponding edge weights file
        eweight = sio.loadmat(edge_weights_files[0])
        eweight = eweight['edgeWeightsMatrix']
        
        # Create the character string variable
        node_weights_string = os.path.basename(node_weights_file)[:14].replace("_", " ")
    else:
        print(f"No corresponding edge weights file found for {node_weights_file}.")

    # Plot the node weights and edge weights
    figure = plot.plot_nweights(eweight, nweight, node_weights_string, labels)

# %% Plot average nodeweights across conditions, W, K, S, P, X

import matplotlib.pyplot as plt
import scipy.io as sio
# Set the figure size
plt.figure(figsize=(17, 12))  # Increase the figure size to 12 inches by 8 inches

nodedata_dir = '/Users/borjan/code/python/AnesthesiaProjectEmergence/results/data'

# Load the label names from the results/utils/ directory
label_names_path = '/Users/borjan/code/python/AnesthesiaProjectEmergence/labels_new.npy'
label_names = np.load(label_names_path)

# Define the file patterns to search for
file_patterns = ['2macro', '6macro', '9macro']

# Initialize empty lists to hold the column vectors and labels
column_vectors = []
column_labels = []

# Define the order of labels
label_order = ['WAKE', 'KET', 'SLEEP', 'XENON', 'PROP']

# Loop over each file pattern
for pattern in file_patterns:
    # Get a list of all files in the directory that contain the pattern and '.mat' in their filenames
    file_list = glob.glob(os.path.join(nodedata_dir, f'*{pattern}*.mat'))
    
    # Initialize empty lists to hold the aligned column vectors and labels
    aligned_column_vectors = []
    aligned_column_labels = []
    
    # Loop over each label in the desired order
    for label in label_order:
        # Find the file corresponding to the label
        matching_files = [file for file in file_list if label in file]
        
        # Check if a matching file is found
        if len(matching_files) > 0:
            # Load the file using scipy.io.loadmat
            data = sio.loadmat(matching_files[0])
            
            if 'avg_nodes' in data:
                # Extract the column vector from the loaded data
                column_vector = data['avg_nodes']
                # Concatenate the column vector to the aligned list
                aligned_column_vectors.append(column_vector)
                # Append the label to the aligned list
                aligned_column_labels.append(label)
            else:
                print(f"Key 'avg_nodes' not found in file: {matching_files[0]}")
        else:
            print(f"No file found for label: {label}")
    
    # Concatenate the aligned column vectors and labels to the main lists
    column_vectors.extend(aligned_column_vectors)
    column_labels.extend(aligned_column_labels)

# Concatenate all the column vectors into a single array
concatenated_data = np.concatenate(column_vectors, axis=0)

# Plot the concatenated data as a heatmap with y-axis tick labels
sns.heatmap(concatenated_data.T, cmap='bone_r', yticklabels=label_names)

# Set the title
plt.title("ROI Contribution to Emergent Communication Subspaces", fontsize=18, fontweight='bold', pad=20)

# Set the x-axis tick labels
plt.xticks(range(len(column_labels)), column_labels, ha='center', fontsize=12, fontweight='bold')
plt.gca().set_xticklabels(column_labels, ha='left', fontsize=12, fontweight='bold')

# Save the figure as SVG and PNG
save_path_svg = '/Users/borjan/code/python/AnesthesiaProjectEmergence/results/heatmaps/roi_contribution_all.svg'
save_path_png = '/Users/borjan/code/python/AnesthesiaProjectEmergence/results/heatmaps/roi_contribution_all.png'

plt.savefig(save_path_svg, format='svg', dpi=600)
plt.savefig(save_path_png, format='png', dpi=600)


# %%
