#%%
import mne 
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import plot
import seaborn as sns





Brain = mne.viz.get_brain_class()

subjects_dir = mne.datasets.sample.data_path() / "subjects"
#mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=True)

#%%
#labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'both', subjects_dir=subjects_dir)
labels_combined = mne.read_labels_from_annot('fsaverage', 'HCPMMP1_combined', 'lh', subjects_dir=subjects_dir)
brain = Brain('fsaverage', 'lh', surf='pial', subjects_dir=subjects_dir,
                cortex='low_contrast', background='white', size=(800, 600),)

brain.add_annotation('HCPMMP1_combined', borders=False)





# %%
directory = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/ssdiData/"
filename_nodeweights = "H0010W_mdim_3_node_weights.mat"
nweight = scipy.io.loadmat(os.path.join(directory, filename_nodeweights)) # load in node weights
nweight = nweight['node_weights'][:,0]    # extract node weights
nweight = nweight.reshape(-1, 1)  # add an extra axis to nweight

# Color the regions based on nweight
label_colour = []
for i, label in enumerate(labels_combined):
    label_colour.append((label.name, plt.cm.viridis(nweight[i])))

    rgb_colors = [label[1][0][:3] for label in label_colour]
    new_variable = np.array(rgb_colors)

# brain.add_data(nweight, colormap='viridis', alpha=0.8)

brain.add_annotation('HCPMMP1_combined', borders=False, color=new_variable, alpha=0.8)



# %%
