
from nilearn import datasets
from nilearn import image as nimg
from nilearn import plotting as nplot

import nibabel as nib

import matplotlib as mpl

#%% import the parcellation atlas

parcels_dir = '../resources/rois/'
atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011(parcels_dir)

#%%

# Define where to slive the image
cut_coords = (8, -4, 9)
# Show the colorbar
colorbar=True
#colour scheme to show when viewing
cmap='Paired'

# plot all parcellation schemas refered to by atlast_yeo_2011
nplot.plot_roi(atlas_yeo_2011['thin_7'], cut_coords=cut_coords, colorbar=colorbar, cmap=cmap, title='thin_7')
nplot.plot_roi(atlas_yeo_2011['thin_17'], cut_coords=cut_coords, colorbar=colorbar, cmap=cmap, title='thin_77')
nplot.plot_roi(atlas_yeo_2011['thick_7'], cut_coords=cut_coords, colorbar=colorbar, cmap=cmap, title='thick_7')

import matplotlib.pyplot as plt
plt.show()

#%%

atlas_yeo = atlas_yeo_2011['thick_7']

func_file = '../data/ds000030/derivatives/fmriprep/sub-10788/func/sub-10788_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
func_img = nib.load(func_file)

