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


#%% 1. Import data from HDD

data_dir = '/Volumes/dataSets/restEEGHealthySubjects/rawData/'
figures_dir = '/Volumes/dataSets/restEEGHealthySubjects/figures/'
preprocessed_dir = '/Volumes/dataSets/restEEGHealthySubjects/preprocessedData/'
subject_file = 'H0010W.mat'
#%%
montage = mne.channels.read_custom_montage('/Volumes/dataSets/restEEGHealthySubjects/nexstim.sfp')

for file in glob(data_dir + '*.mat'):



    # intialise subject_file
    # subject_ext = file[49]
    subject_file = file[49:54]
    subject_cond = file[54]

    #%% 2. CLEAN DATA (this might not need to be here as the dataset should be saved from _psd_sensor_space.py_
    rawDataStructure = read_mat(file)
    #rawDataStructure = read_mat(os.path.join(data_dir, subject_file))
    info = mne.create_info(ch_names=rawDataStructure['chlocs']['labels'], sfreq=rawDataStructure['srate'], ch_types='eeg')
    info.set_montage(montage)
    raw = mne.io.RawArray(rawDataStructure['EEG'].T, info=info)    # create Raw data structure
    raw_filtered = raw.copy().filter(l_freq=1, h_freq=150)
    raw_notched = raw_filtered.copy().notch_filter(freqs=[50, 100], method='spectrum_fit')
    raw_downsample = raw_notched.copy().resample(sfreq=256)
    raw_avg_ref = raw_downsample.copy().set_eeg_reference(ref_channels='average')
    raw_downsample.set_eeg_reference(projection=True)

    #%% 3. LOADING IN FSAVERAGE FOR FORWARD-OPERATOR _fwd_

    fs_dir = fetch_fsaverage(verbose=True)
    fs_subjects_dir = os.path.dirname(fs_dir)
    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    #src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    # use the below source reconstruction, as it uses less source dipoles.
    src_downsampled = mne.setup_source_space(subject, spacing='ico4', subjects_dir=fs_subjects_dir, n_jobs=-1)
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


    # Check that the locations of EEG electrodes is correct with respect to MRI
    #mne.viz.plot_alignment(
    #    raw_avg_ref.info, src=src, eeg=['original', 'projected'], trans=trans,
    #    show_axes=True, mri_fiducials=True, dig='fiducials')

    #%% 4. COMPUTE COVARIANCE MATRIX for lead field construction (gain matrix, or spatial filter).

    # 4.1. create 2 second epochs from continuous data

    epochs = mne.make_fixed_length_epochs(raw_downsample, duration=2) #can switch back to preload=True, if needed

    # 4.2. covariance matrix of sensor space data

    data_cov = mne.compute_covariance(epochs, method='empirical')
    data_cov.plot(epochs.info)

    #%% 5. COMPUTE FORWARD SOLUTION

    fwd = mne.make_forward_solution(raw_downsample.info, trans=trans, src=src_downsampled,
                                    bem=bem, eeg=True, mindist=5.0, n_jobs=-1)
    print(fwd)

    sourcespace = fwd['src']


    #%% 6. COMPUTING SPATIAL FILTER

    filters = mne.beamformer.make_lcmv(epochs.info, fwd, data_cov, reg=0.05,
                        pick_ori='max-power',
                        weight_norm='unit-noise-gain', rank=None)

    #%% 7. COMPUTING SOURCE TIME-COURSE!

    stc = mne.beamformer.apply_lcmv_epochs(epochs, filters)

    #%% 8. APPLY PARCELLATION

    labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1_combined', subjects_dir=fs_subjects_dir) # HCPMMP1_combined has 46 regions.
    label_names = [label.name for label in labels]

    # Create the results/utils/ directory if it doesn't exist
    save_dir = os.path.join(os.getcwd(), 'results', 'utils')
    os.makedirs(save_dir, exist_ok=True)

    # Save label_names.npy in the results/utils/ directory
    np.save(os.path.join(save_dir, 'label_names.npy'), label_names)
    
    label_colors = [label.color for label in labels]

    # Average the source estimates within each label using sign-flips to reduce
    # signal cancellations, also here we return a generator

    # *src*  needs to be the source space information and now just the path to the file as set above
    label_ts = mne.extract_label_time_course(stc, labels, sourcespace, mode='mean_flip', return_generator=False, allow_empty=True)

    # save label_ts: these are the time-courses of the source-localised data.

    np.save(os.path.join(preprocessed_dir, "{}{}_source_time_series".format(subject_file, subject_cond)), label_ts)
    sio.savemat("/Volumes/dataSets/restEEGHealthySubjects/preprocessedData/sourceReconstructions/{}{}_source_time_series.mat".format(subject_file, subject_cond), {"source_ts": label_ts})




