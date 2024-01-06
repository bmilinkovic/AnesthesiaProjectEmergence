#%%
import time as tm
import os
from glob import glob


import pandas as pd
import numpy as np
import scipy as sc
import scipy.io as sio
from scipy import signal

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import mne
from mne.datasets import fetch_fsaverage
from pymatreader import read_mat
import os

#%% Import data from HDD

data_dir = '/Volumes/dataSets/restEEGHealthySubjects/rawData/'
figures_dir = '/Volumes/dataSets/restEEGHealthySubjects/figures/'
preprocessed_dir = '/Volumes/dataSets/restEEGHealthySubjects/preprocessedData/average_powerspec/'

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(preprocessed_dir, exist_ok=True)
#subject_file = 'H0010W.mat'

montage = mne.channels.read_custom_montage('/Volumes/dataSets/restEEGHealthySubjects/nexstim.sfp')
powerspec = []

#%% 
for file in glob(data_dir + '*S*.mat'):
    # intialise subject_file
    # subject_ext = file[49]
    subject_file = file[49:54]
    subject_cond = file[54]

    rawDataStructure = read_mat(file)
    info = mne.create_info(ch_names=rawDataStructure['chlocs']['labels'], sfreq=rawDataStructure['srate'], ch_types='eeg')
    info.set_montage(montage)
    raw = mne.io.RawArray(rawDataStructure['EEG'].T, info=info)    # create Raw data structure
    raw_filtered = raw.copy().filter(l_freq=1, h_freq=150)
    raw_notched = raw_filtered.copy().notch_filter(freqs=[50, 100], method='spectrum_fit')
    raw_downsample = raw_notched.copy().resample(sfreq=256)
    raw_avg_ref = raw_downsample.copy().set_eeg_reference(ref_channels='average')

    #SENSOR SPACE ANALYSIS

    # POWER SPECTRAL DENSITY WITH GIVEN FREQUENCIES
    psd = raw_avg_ref.compute_psd(method='welch', fmax=40)
    psds, freqs = psd.get_data(return_freqs=True)
    powerspec.append(psds)

    # psd_plot = psd.plot()
    # psd.plot_topo(color='r', fig_facecolor='w', axis_facecolor='w')

    fig, ax = plt.subplots()
    raw_avg_ref.plot_psd(ax=ax, average=True, show=False, color='red')
    ax.set_title('Power-Spectral Density (PSD) of Sleep condition', fontweight='bold', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'Sleep_sensor_PSD.svg'))
    # fig.show()
#%% save powerspec
np.save(os.path.join(figures_dir, "powerspec_sensor"), powerspec)

#%% plotting for averages

# def plot_output(psda, psdb, freqs, cond_a="high", cond_b="low"):

#   cond = [psda, psdb]
#   labellist = [cond_a, cond_b]
#   log_both = []
  
#   for index, c in enumerate(cond):
      
#       # log transform your data 
#       log = [np.log10(a) for a in c]

#       #zscore
#       #log = [cb - np.mean(cb, axis=0) / np.std(cb, axis=0) for cb in c]
      
#       # mean over epochs and channels
#       log_all = [np.mean(a[:, :, :], axis=(0,1)) for a in log]

#       #save PSD per subject
#       log_both.append(log_all)

#       #mean over subjects
#       log_all = np.mean(np.array(log_all), axis=0)
#       high_sd = np.std(np.array(log_all), axis=0)
#       plt.plot(freqs, log_all.T, label=labellist[index])

#       #plot standard deviation as shaded area 
#       #plt.fill_between(freqs, log_all.T - high_sd, log_all.T + high_sd, alpha=0.5)
#       plt.legend()
  
#   plt.show(block=False)
#   os.chdir(save_dir)
#   plt.savefig("psd_mt_Fig2_confhit.svg", dpi=600, bbox_inches='tight', transparent=True)

#%% load fs-average head model

fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)
# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    mneRawEEGArray.info, src=src, eeg=['original', 'projected'], trans=trans,
    show_axes=True, mri_fiducials=True, dig='fiducials')

#%% We need the covariance matrix and the forward soltuon to create our lead field matrix.


# create 2 second epochs from continuous data

epochs = mne.make_fixed_length_epochs(mneRawEEGArray, duration=2, preload=False)
event_related_plot = epochs.plot_image(picks=['EEG43'])

# 1. covariance matrix of sensor space data

data_cov = mne.compute_covariance(epochs, method='empirical')
data_cov.plot(epochs.info)

# 2. forward solution
fwd = mne.make_forward_solution(mneRawEEGArray.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=-1)
print(fwd)

# computing the lcmv filter

filters = mne.beamformer.make_lcmv(epochs.info, fwd, data_cov, reg=0.05,
                    pick_ori='max-power',
                    weight_norm='unit-noise-gain', rank=None)

# apply filter to get source activity

stc = mne.beamformer.apply_lcmv_epochs(epochs, filters)







