#%%
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind, f_oneway

#%%
directory = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/ssdiData/"
#data = scipy.io.loadmat(os.path.join(directory, 'H0010P_mdim_2_dynamical_dependence.mat'))

#%% PLOTTING VIOLIN PLOTS OF DD VALUES FOR EACH CONDITION

fig, ax = plt.subplots()

dynamical_dependence_data = []
labels = []

# Define the colors for each label
colors = {'W': 'green', 'K': 'purple', 'S': 'red', 'X': 'gray', 'P': 'blue'}

for filename in os.listdir(directory):
        if filename.endswith("mdim_2_dynamical_dependence.mat"):
                # Load the data file
                data = scipy.io.loadmat(os.path.join(directory, filename))

                # Append the data as column vectors
                dynamical_dependence_data.append(data['dopto'][0][:])

                # Get the first 6 characters of the file name
                label = filename[5:6]
                labels.append(label)

                # Add the violin plot with the color
                plots = ax.violinplot(data['dopto'][0][:], positions=[len(dynamical_dependence_data)], showmeans=True, showmedians=False, showextrema=False)
                for pc in plots['bodies']:
                        pc.set_facecolor(colors[label])

                # Set the color of the mean lines
                plots['cmeans'].set_color(colors[label])

# Create a legend with the same labels as the color dictionary
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=colors[label], markersize=10) for label in colors.keys()]
ax.legend(handles=legend_elements)


# Set the y-axis label
ax.set_ylabel('Dynamical Dependence', fontsize=12, fontweight='bold')

# Set the x-axis label
ax.set_xlabel('Participant Number', fontsize=12, fontweight='bold')

# Set title

ax.set_title('Dynamical dependence across conditions', fontsize=12, fontweight='bold', pad=16)



#%% GRAND AVERAGE DD values across conditions.

fig, ax = plt.subplots()

dynamical_dependence_data = []
labels = []

# Define the colors for each label
colors = {'W': 'green', 'K': 'purple', 'S': 'red', 'X': 'gray', 'P': 'blue'}

# Create a dictionary to store the concatenated data for each label
concatenated_data = {}

# Define the desired order of labels
desired_order = ['W', 'K', 'S', 'X', 'P']

for filename in os.listdir(directory):
        if filename.endswith("mdim_11_dynamical_dependence.mat"):
                # Load the data file
                data = scipy.io.loadmat(os.path.join(directory, filename))

                # Get the first 6 characters of the file name
                label = filename[5:6]

                # Append the data to the corresponding label in the dictionary
                if label not in concatenated_data:
                        concatenated_data[label] = data['dopto'][0][:]
                else:
                        concatenated_data[label] = np.concatenate((concatenated_data[label], data['dopto'][0][:]))

# Sort the concatenated data and labels based on the desired order
sorted_data = []
sorted_labels = []
for label in desired_order:
        if label in concatenated_data:
                sorted_data.append(concatenated_data[label])
                sorted_labels.append(label)

# Plot the grand violin plot for each label
for label, data in zip(sorted_labels, sorted_data):
        # Append the data to the overall dynamical_dependence_data list
        dynamical_dependence_data.append(data)

        # Add the violin plot with the color
        plots = ax.violinplot(data, positions=[len(dynamical_dependence_data)], showmeans=True, showmedians=False, showextrema=False)
        for pc in plots['bodies']:
                pc.set_facecolor(colors[label])

        # Set the color of the mean lines
        plots['cmeans'].set_color(colors[label])

# Create a legend with the same labels as the color dictionary
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=colors[label], markersize=10) for label in colors.keys()]
ax.legend(handles=legend_elements)

# Set title
ax.set_title('11-Communication Subspace', fontsize=12, fontweight='bold', pad=16)

# Set the y-axis label
ax.set_ylabel('Dynamical Dependence', fontsize=12, fontweight='bold')

# Set the x-axis label
ax.set_xlabel('Condition', fontsize=12, fontweight='bold')

# Set the x-axis tick locations
ax.set_xticks(range(1, len(sorted_labels) + 1))

# Set the x-axis tick labels
ax.set_xticklabels(sorted_labels)



fig_directory = './results/dd_EEG/'
if not os.path.exists(fig_directory):
        os.makedirs(fig_directory)

fig.savefig(os.path.join(fig_directory, 'dd_mdim_11_1.png'), dpi=600, bbox_inches='tight')


#%% Run T-Test and F-test across conditions
import itertools
from scipy.stats import ttest_ind, f_oneway

# Generate all combinations of indices
index_combinations = list(itertools.combinations(range(len(dynamical_dependence_data)), 2))

# Perform t-test and f-test for each combination
for combination in index_combinations:
        index1, index2 = combination
        data1 = dynamical_dependence_data[index1]
        data2 = dynamical_dependence_data[index2]
        
        condition1 = labels[index1]
        condition2 = labels[index2]

        # Perform t-test
        t_statistic, p_value = ttest_ind(data1, data2)
        print(f"T-Test Results for combination {condition1} and {condition2}:")
        print("T-Statistic:", t_statistic)
        print("P-Value:", p_value)

        # Perform f-test
        f_statistic, p_value = f_oneway(data1, data2)
        print(f"F-Test Results for combination {condition1} and {condition2}:")
        print("F-Statistic:", f_statistic)
        print("P-Value:", p_value)


#%% Plotting the node weights of the whole brain network


directory = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/ssdiData/"
filename_nodeweights = "H0905X_mdim_18_node_weights.mat"

filename_edgeweights = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/pwcgc_matrix/pwcgc_matrix_H0905X_source_time_series_34-of-34.mat"

nweight = scipy.io.loadmat(os.path.join(directory, filename_nodeweights)) # load in node weights
nweight = nweight['node_weights']    # extract node weights

eweight = scipy.io.loadmat(filename_edgeweights) # load in edge weights
eweight = eweight['edgeWeightsMatrix']    # extract edge weights

#%%
def plot_nweights_wholebrain(eweights, nweights, macrosize, opt_number):
        subset = pd.DataFrame(eweights)
        # subset.columns = ['0', '1', '2']
        # subset.index = ['0','1','2']
        G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
        G.remove_edges_from(list(nx.selfloop_edges(G)))                    # remove self edges
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) # Extracts the edges and their corresponding weights into different tuples

        # PLOTTING THE PWCGC MATRIX, GRAPH AND MACRO PROJECTION ON GRAPH.
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(nrows=1, ncols=1)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("{0}-Macro on GC-graph of coupled {1}-node model".format(int(macrosize), int(len(subset))), fontsize=12, fontweight='bold', pad=16)
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color=nweights[:,opt_number], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black') # nweights[:,0] will plot the optimal projection of the first macro variable on the graph.
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=800, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
        edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])

        return fig

# %%
figure = plot_nweights_wholebrain(eweight, nweight, 18, 0)
plt.show()
# %%
