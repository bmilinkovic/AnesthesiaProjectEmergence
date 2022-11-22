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

#%% Import data from HDD

data_dir = '/Volumes/dataSets/restEEGHealthySubjects/rawData/'
figures_dir = '/Volumes/dataSets/restEEGHealthySubjects/figures/'
preprocessed_dir = '/Volumes/dataSets/restEEGHealthySubjects/preprocessedData/'
#subject_file = 'H0010W.mat'

montage = mne.channels.read_custom_montage('/Volumes/dataSets/restEEGHealthySubjects/nexstim.sfp')
powerspec = []

for file in glob(data_dir + '*.mat'):
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

    #%% SENSOR SPACE ANALYSIS

    # POWER SPECTRAL DENSITY WITH GIVEN FREQUENCIES
    psd = raw_avg_ref.compute_psd(method='welch', fmax=128)
    psds, freqs = psd.get_data(return_freqs=True)
    powerspec.append(psds)

    # psd_plot = psd.plot()
    # psd.plot_topo(color='r', fig_facecolor='w', axis_facecolor='w')

    fig, ax = plt.subplots()
    raw_avg_ref.plot_psd(ax=ax, show=False)
    ax.set_title('Power-Spectral Density (PSD) of Subject {} in {}-condition'.format(subject_file, subject_cond), fontweight='bold', fontsize=10)
    fig.tight_layout()
    plt.savefig(os.path.join(figures_dir, '{}_sensor_PSD.svg'.format(file[49:55])))
    # fig.show()

np.save(os.path.join(figures_dir, "powerspec_sensor"), powerspec)

#%% load fs-average head model
#
# fs_dir = fetch_fsaverage(verbose=True)
# subjects_dir = os.path.dirname(fs_dir)
# # The files live in:
# subject = 'fsaverage'
# trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
# src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
# bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
#
#
# # Check that the locations of EEG electrodes is correct with respect to MRI
# mne.viz.plot_alignment(
#     mneRawEEGArray.info, src=src, eeg=['original', 'projected'], trans=trans,
#     show_axes=True, mri_fiducials=True, dig='fiducials')
#
# #%% We need the covariance matrix and the forward soltuon to create our lead field matrix.
#
#
# # create 2 second epochs from continuous data
#
# epochs = mne.make_fixed_length_epochs(mneRawEEGArray, duration=2, preload=False)
# event_related_plot = epochs.plot_image(picks=['EEG43'])
#
# # 1. covariance matrix of sensor space data
#
# data_cov = mne.compute_covariance(epochs, method='empirical')
# data_cov.plot(epochs.info)
#
# # 2. forward solution
# fwd = mne.make_forward_solution(mneRawEEGArray.info, trans=trans, src=src,
#                                 bem=bem, eeg=True, mindist=5.0, n_jobs=-1)
# print(fwd)
#
# # computing the lcmv filter
#
# filters = mne.beamformer.make_lcmv(epochs.info, fwd, data_cov, reg=0.05,
#                     pick_ori='max-power',
#                     weight_norm='unit-noise-gain', rank=None)
#
# # apply filter to get source activity
#
# stc = mne.beamformer.apply_lcmv_epochs(epochs, filters)







