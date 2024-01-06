import os
import time
import datetime

import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import networkx as nx
import pandas as pd
from scipy import stats

def plot_gc(edge_weights, filename, labels):
    """
    Plots a Granger-causal matrix and graph based on the given edge weights.

    Parameters:
    - edge_weights (list): A list of lists containing the edge weights.
    - filename (str): The filename for saving the plot.
    - labels (numpy.ndarray): The labels for each node.

    Returns:
    - fig (matplotlib.figure.Figure): The resulting figure object.
    """
    # the range is over the amount of simulations performed.
    subset = pd.DataFrame(edge_weights)
    # subset.columns = ['0', '1', '2']
    # subset.index = ['0','1','2']
    G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
    G.remove_edges_from(list(nx.selfloop_edges(G)))                    # remove self edges
    #node_labels = {i: labels[i] for i in range(len(labels))}
    #nx.relabel_nodes(G, node_labels, copy=False)
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) # Extracts the edges and their corresponding weights into different tuples

    n_cols = 2
    n_rows = 1
    # Calculate the width and height ratios
    width_ratios = [1.2] * n_cols
    height_ratios = [1] * n_rows
    # PLOTTING THE PWCGC MATRIX and GRAPH.
    fig = plt.figure(figsize=(22.44, 10.67)) # Increase the figsize by 1/3
    gs = GridSpec(nrows=n_rows, ncols=n_cols, width_ratios=width_ratios, height_ratios=height_ratios)

    # Sort the rows and columns based on the tick-labels starting with 'l' and 'r'
    row_order = sorted(range(len(labels)), key=lambda i: (labels[i][0], labels[i]))
    col_order = sorted(range(len(labels)), key=lambda i: (labels[i][0], labels[i]))

    # Rearrange the subset matrix based on the sorted row and column order
    subset = subset.iloc[row_order, col_order]

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title(f"{filename[125:131]} {int(len(subset))}-region Granger-causal matrix", fontsize=20, fontweight='bold', pad=26)
    mask = subset == 0
    sns.heatmap(subset, cmap=mpl.cm.bone_r, center=0.5, linewidths=.6, linecolor='black', annot=False, cbar_kws={'label': 'G-causal estimate values', 'shrink': 0.5, 'orientation': 'vertical'}, ax=ax0, mask=mask, xticklabels=[labels[i] for i in col_order], yticklabels=[labels[i] for i in row_order])

    cbar = ax0.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('G-causal estimate values', fontsize=14, fontweight='bold', labelpad=10)

    # Modify the x and y tick labels to start from 1 instead of 0
    ax0.set_xticks(np.arange(len(labels)) + 0.5)
    ax0.set_yticks(np.arange(len(labels)) + 0.5)
    ax0.set_xticklabels([labels[i] for i in col_order], rotation=90)
    ax0.set_yticklabels([labels[i] for i in row_order], rotation=0)
    ax0.set_xlabel('To', fontsize=16, fontweight='bold',)
    ax0.set_ylabel('From', fontsize=16, fontweight='bold', labelpad=10)

    ax0.set_aspect('equal')
    ax0.tick_params(labelsize=10) # Set the tick_params label size to 12

    ax1 = fig.add_subplot(gs[0, 1])
    #ax1.set_title(f"{filename[125:131]} {int(len(subset))}-region Granger-causal graph", fontsize=12, fontweight='bold', pad=16)
    ax1.set_title('')
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightgray', linewidths=1.0, edgecolors='black', ax=ax1) # Change the node_size to adjust the size of the nodes
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=800, width=2.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r, ax=ax1) # Change the node_size to adjust the size of the nodes
    nx.draw_networkx_labels(G, pos, labels={i: labels[i] for i in range(len(labels))}, font_size=11, font_family="helvetica", ax=ax1) # Use the provided labels for node labels
    # Create a legend for the node labels
    # legend_labels = {i: labels[i] for i in range(len(labels))}
    # legend_handles = [mpl.patches.Patch(label=label) for label in legend_labels.values()]
    # ax1.legend(handles=legend_handles, loc='upper right', fontsize=10)
    ax1.set_aspect('equal')
    ax1.tick_params(labelsize=11)

    ax1.axis('off')

    plt.tight_layout()
    # Save the figure as an eps and a png file with the name corresponding to the size of the granger causal matrix
    if not os.path.exists('results'):
        os.makedirs('results')

    filename = f"{filename[112:131]}_plot"
    fig.savefig(os.path.join(os.getcwd(), 'results', f"{filename}.svg"), format='svg')
    fig.savefig(os.path.join(os.getcwd(), 'results', f"{filename}.png"), format='png')

    return fig


def plot_gc_ave(edge_weights, filename, labels):
    """
    Plots a Granger-causal matrix and graph based on the given edge weights.

    Parameters:
    - edge_weights (list): A list of lists containing the edge weights.
    - filename (str): The filename for saving the plot.
    - labels (numpy.ndarray): The labels for each node.

    Returns:
    - fig (matplotlib.figure.Figure): The resulting figure object.
    """
    # the range is over the amount of simulations performed.
    subset = pd.DataFrame(edge_weights)
    # subset.columns = ['0', '1', '2']
    # subset.index = ['0','1','2']
    G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
    G.remove_edges_from(list(nx.selfloop_edges(G)))                    # remove self edges
    #node_labels = {i: labels[i] for i in range(len(labels))}
    #nx.relabel_nodes(G, node_labels, copy=False)
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) # Extracts the edges and their corresponding weights into different tuples

    n_cols = 2
    n_rows = 1
    # Calculate the width and height ratios
    width_ratios = [1.2] * n_cols
    height_ratios = [1] * n_rows
    # PLOTTING THE PWCGC MATRIX and GRAPH.
    fig = plt.figure(figsize=(22.44, 10.67)) # Increase the figsize by 1/3
    gs = GridSpec(nrows=n_rows, ncols=n_cols, width_ratios=width_ratios, height_ratios=height_ratios)

    # Sort the rows and columns based on the tick-labels starting with 'l' and 'r'
    row_order = sorted(range(len(labels)), key=lambda i: (labels[i][0], labels[i]))
    col_order = sorted(range(len(labels)), key=lambda i: (labels[i][0], labels[i]))

    # Rearrange the subset matrix based on the sorted row and column order
    subset = subset.iloc[row_order, col_order]

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title(f"{filename} Granger-causal matrix", fontsize=20, fontweight='bold', pad=26)
    mask = subset == 0
    center_value = np.nanmean(subset.values)  # Calculate the average value of the subset matrix while excluding NaN values
    sns.heatmap(subset, cmap=mpl.cm.bone_r, center=center_value, linewidths=.6, linecolor='black', annot=False, cbar_kws={'label': 'G-causal estimate values', 'shrink': 0.5, 'orientation': 'vertical'}, ax=ax0, mask=mask, xticklabels=[labels[i] for i in col_order], yticklabels=[labels[i] for i in row_order])

    cbar = ax0.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('G-causal estimate values', fontsize=14, fontweight='bold', labelpad=10)

    # Modify the x and y tick labels to start from 1 instead of 0
    ax0.set_xticks(np.arange(len(labels)) + 0.5)
    ax0.set_yticks(np.arange(len(labels)) + 0.5)
    ax0.set_xticklabels([labels[i] for i in col_order], rotation=90)
    ax0.set_yticklabels([labels[i] for i in row_order], rotation=0)
    ax0.set_xlabel('To', fontsize=16, fontweight='bold',)
    ax0.set_ylabel('From', fontsize=16, fontweight='bold', labelpad=10)

    ax0.set_aspect('equal')
    ax0.tick_params(labelsize=10) # Set the tick_params label size to 12

    ax1 = fig.add_subplot(gs[0, 1])
    #ax1.set_title(f"{filename[125:131]} {int(len(subset))}-region Granger-causal graph", fontsize=12, fontweight='bold', pad=16)
    ax1.set_title('')
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightgray', linewidths=1.0, edgecolors='black', ax=ax1) # Change the node_size to adjust the size of the nodes
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=800, width=2.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r, ax=ax1) # Change the node_size to adjust the size of the nodes
    nx.draw_networkx_labels(G, pos, labels={i: labels[i] for i in range(len(labels))}, font_size=11, font_family="helvetica", ax=ax1) # Use the provided labels for node labels
    # Create a legend for the node labels
    # legend_labels = {i: labels[i] for i in range(len(labels))}
    # legend_handles = [mpl.patches.Patch(label=label) for label in legend_labels.values()]
    # ax1.legend(handles=legend_handles, loc='upper right', fontsize=10)
    ax1.set_aspect('equal')
    ax1.tick_params(labelsize=11)

    ax1.axis('off')

    plt.tight_layout()
    # Save the figure as an eps and a png file with the name corresponding to the size of the granger causal matrix
    if not os.path.exists('results'):
        os.makedirs('results')

    fig.savefig(os.path.join(os.getcwd(), 'results', f"{filename}.svg"), format='svg')
    fig.savefig(os.path.join(os.getcwd(), 'results', f"{filename}.png"), format='png')

    return fig

def plot_nweights(eweights, nweights, node_file_string, labels, opt_number=0):
    """
    Plots the PWCGC matrix, graph, and macro projection on graph.

    Args:
    - eweights (numpy.ndarray): The edge weights of the model.
    - nweights (numpy.ndarray): The node weights of the model.
    - macrosize (int): The size of the macro variable.
    - opt_number (int): The optimal projection of the macro variable on the graph.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
        
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
    ax0.set_title(f"{node_file_string} emergent communication subspace", fontsize=12, fontweight='bold', pad=16)
    #ax0.set_title('')
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=nweights[:,opt_number], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black') # nweights[:,0] will plot the optimal projection of the first macro variable on the graph.
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=800, width=2.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r, ax=ax0)
    nx.draw_networkx_labels(G, pos, labels={i: labels[i] for i in range(len(labels))}, font_size=11, font_family="helvetica", ax=ax0) # Use the provided labels for node labels
    edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])

    # Create a legend for the node labels
    # legend_labels = {i: labels[i] for i in range(len(labels))}
    # legend_handles = [mpl.patches.Patch(label=label) for label in legend_labels.values()]
    # ax1.legend(handles=legend_handles, loc='upper right', fontsize=10)
    ax0.set_aspect('equal')
    ax0.tick_params(labelsize=11)

    node_file_string_save = node_file_string.replace(" ", "_")
    # Save the figure as an eps and a png file in a directory called "results" in the current working directory. If the directory doesn't exist, create it.
    if not os.path.exists('results/macros/'):
        os.makedirs('results/macros/')
    filename = f"{node_file_string_save}_macro_on_graph"
    fig.savefig(os.path.join(os.getcwd(), 'results/macros/', f"{filename}.svg"), format='svg')
    fig.savefig(os.path.join(os.getcwd(), 'results/macros/', f"{filename}.png"), format='png')

    return fig


def plot_nweights_avg(eweights_avg, nweights_avg, node_file_string, labels, opt_number=0):
    """
    Plots the PWCGC matrix, graph, and macro projection on graph.

    Args:
    - eweights (numpy.ndarray): The edge weights of the model.
    - nweights (numpy.ndarray): The node weights of the model.
    - macrosize (int): The size of the macro variable.
    - opt_number (int): The optimal projection of the macro variable on the graph.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
        
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
    ax0.set_title(f"{node_file_string} emergent communication subspace", fontsize=12, fontweight='bold', pad=16)
    #ax0.set_title('')
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=nweights[:,opt_number], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black') # nweights[:,0] will plot the optimal projection of the first macro variable on the graph.
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=800, width=2.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r, ax=ax0)
    nx.draw_networkx_labels(G, pos, labels={i: labels[i] for i in range(len(labels))}, font_size=11, font_family="helvetica", ax=ax0) # Use the provided labels for node labels
    edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])

    # Create a legend for the node labels
    # legend_labels = {i: labels[i] for i in range(len(labels))}
    # legend_handles = [mpl.patches.Patch(label=label) for label in legend_labels.values()]
    # ax1.legend(handles=legend_handles, loc='upper right', fontsize=10)
    ax0.set_aspect('equal')
    ax0.tick_params(labelsize=11)

    node_file_string_save = node_file_string.replace(" ", "_")
    # Save the figure as an eps and a png file in a directory called "results" in the current working directory. If the directory doesn't exist, create it.
    if not os.path.exists('results/macros/'):
        os.makedirs('results/macros/')
    filename = f"{node_file_string_save}_macro_on_graph"
    fig.savefig(os.path.join(os.getcwd(), 'results/macros/', f"{filename}.svg"), format='svg')
    fig.savefig(os.path.join(os.getcwd(), 'results/macros/', f"{filename}.png"), format='png')

    return fig


def plot_opto(opthist, macro):
    """
    Plots the optimization history and local-optima distances.

    Parameters:
    opthist (list): A list of tuples containing the optimization history for each run.
    optdist (dict): A dictionary containing the local-optima distances for each run.

    Returns:
    fig (matplotlib.figure.Figure): The figure object containing the plotted graphs.
    """

    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(nrows=1, ncols=1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(f'{macro}-Macro Optimisation History: ', fontweight='bold', fontsize=18)
    ax1.set_ylabel('Dynamical Dependence', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Optimisation run', fontweight='bold', fontsize=14)
    for i in range(len(opthist)):
        ax1 = sns.lineplot(data=opthist, legend=False, dashes=False, palette='bone_r')

    # Save the figure as an eps and a png file with the name corresponding to the size of the granger causal matrix
    if not os.path.exists('results/dd/svg'):
        os.makedirs('results/dd/svg')

    if not os.path.exists('results/dd/png'):
        os.makedirs('results/dd/png')

    filename = f"{filename[108:122]}_opto_plot"
    fig.savefig(os.path.join(os.getcwd(), 'results/dd/', f"{filename}.svg"), format='svg')
    fig.savefig(os.path.join(os.getcwd(), 'results/dd/', f"{filename}.png"), format='png')



def plot_opt_dist(optdist, filename):
    fig = plt.figure(figsize=(10,10))

    macro = filename[120:122]
    cmap = sns.color_palette("bone_r", as_cmap=True)
    ax2 = fig.add_subplot(111)
    ax2.set_title(f'{macro}-Macro Optimisation History: ', fontweight='bold', fontsize=18)
    ax2 = sns.heatmap(optdist, cmap=cmap, center=np.max(optdist)/2, cbar_kws={'label': 'Othogonality of subspaces'})
    ax2.set_xlabel('Optimisation run', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Optimisation run', fontweight='bold', fontsize=14)
    ax2.set_xticklabels(ax2.get_xmajorticklabels(), fontsize = 8)
    ax2.set_yticklabels(ax2.get_ymajorticklabels(), fontsize = 8)
    ax2.invert_yaxis()

    # Save the figure as an eps and a png file with the name corresponding to the size of the granger causal matrix
    if not os.path.exists('results/opt/'):
        os.makedirs('results/opt/')

    filename = f"{filename[108:122]}_dist_plot"
    fig.savefig(os.path.join(os.getcwd(), 'results/opt/', f"{filename}.eps"), format='eps')
    fig.savefig(os.path.join(os.getcwd(), 'results/opt/', f"{filename}.png"), format='png')

    return fig

def plot_preopt_dist(preoptdist, filename):
    fig = plt.figure(figsize=(10,10))

    macro = filename[120:122]
    cmap = sns.color_palette("bone_r", as_cmap=True)
    ax1 = fig.add_subplot(111)
    #ax1.set_title(f'{macro} - Macro similarity matrix ', fontweight='bold', fontsize=18)
    ax1.set_title('') # <-- set the title of the figure to empty string
    ax1 = sns.heatmap(preoptdist, cmap=cmap, center=np.max(preoptdist)/2, cbar=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    # ax2 = sns.heatmap(preoptdist, cmap=cmap, center=np.max(preoptdist)/2, cbar_kws={'label': 'Othogonality of subspaces'})
    # ax2.set_xlabel('Optimisation run', fontweight='bold', fontsize=14)
    # ax2.set_ylabel('Optimisation run', fontweight='bold', fontsize=14)
    # ax2.set_xticklabels(ax2.get_xmajorticklabels(), fontsize = 8)
    # ax2.set_yticklabels(ax2.get_ymajorticklabels(), fontsize = 8)
    ax1.invert_yaxis()

    # Save the figure as an eps and a png file with the name corresponding to the size of the granger causal matrix

    # filename = f"{filename[108:121]}_preopt_similarity_matrix_plot"
    # fig.savefig(os.path.join(os.getcwd(), 'results/preopt/', f"{filename}.svg"), format='svg')
    # fig.savefig(os.path.join(os.getcwd(), 'results/preopt/', f"{filename}.png"), format='png')

    return fig