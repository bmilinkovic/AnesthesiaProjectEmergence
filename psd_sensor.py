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
import numpy as np


#%%
data_dir = '/Volumes/dataSets/restEEGHealthySubjects/rawData/'
figures_dir = '/Volumes/dataSets/restEEGHealthySubjects/figures/'
preprocessed_dir = '/Volumes/dataSets/restEEGHealthySubjects/preprocessedData/average_powerspec/'

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(preprocessed_dir, exist_ok=True)

montage = mne.channels.read_custom_montage('/Volumes/dataSets/restEEGHealthySubjects/nexstim.sfp')
powerspec = [] # <-- initialize empty list to store powerspecs

# #%%

# for file in glob(data_dir + '*S*.mat'):
#     raw_data = read_mat(file)
#     info = mne.create_info(ch_names=raw_data['chlocs']['labels'], sfreq=raw_data['srate'], ch_types='eeg')
#     info.set_montage(montage)
#     raw = mne.io.RawArray(raw_data['EEG'].T, info=info)    # create Raw data structure
#     raw_filtered = raw.filter(l_freq=1, h_freq=150).notch_filter(freqs=[50, 100], method='spectrum_fit').resample(sfreq=1024).set_eeg_reference(ref_channels='average')
#     psd = raw_filtered.compute_psd(method='welch', fmin=2, fmax=40)
#     psds, freqs = psd.get_data(return_freqs=True)
    
#     # Calculate psd_new using list comprehension
#     # psd_new = [(p**2) / f for p, f in zip(psds[3:], freqs[3:])]
#     # psd_new = np.mean(psd_new, axis=0) # <-- average over channels
    
#     powerspec.append(psds)
#     #powerspec = np.mean(powerspec, axis=0) # <-- average over subjects
  

#     # plt.figure()
#     # ax = plt.axes()
#     # ax.set_title('Grand Average PSD in Wake condition', fontweight='bold', fontsize=16)
#     # psd.plot(average=True, color='green', alpha=0.3, linewidth=2)
#     # plt.savefig(os.path.join(figures_dir, 'GA_wake_sensor_PSD.svg'))

# #%% save powerspec
# np.save(os.path.join(figures_dir, "all_sleep_powerspec_sensor"), powerspec)
# np.save(os.path.join(figures_dir, "all_sleep_freqs"), freqs)

#%%
# Load powerspec
wake_powerspec = np.load(os.path.join(figures_dir, "all_wake_powerspec_sensor.npy"))
sleep_powerspec = np.load(os.path.join(figures_dir, "all_sleep_powerspec_sensor.npy"))
ket_powerspec = np.load(os.path.join(figures_dir, "all_ket_powerspec_sensor.npy"))
prop_powerspec = np.load(os.path.join(figures_dir, "all_prop_powerspec_sensor.npy"))
xenon_powerspec = np.load(os.path.join(figures_dir, "all_xenon_powerspec_sensor.npy"))
freqs = np.load(os.path.join(figures_dir, "all_wake_freqs.npy"))

# #%% plotting the average

def plot_output(psda, psdb, psdc, psdd, psde, freqs, cond_a="Wake", cond_b="Sleep", cond_c="Ket", cond_d="Prop", cond_e="Xenon"):
    cond = [psda, psdb, psdc, psdd, psde]
    labellist = [cond_a, cond_b, cond_c, cond_d, cond_e]
    log_both = []
  
    colors = ['green', 'red', 'purple', 'blue', 'black']
    line_alpha = 0.7
    fill_alpha = 0.14

    for index, c in enumerate(cond):
        log = np.log10(c)
        log_freqs = np.log10(freqs)

        log_all_channels = np.mean(np.array(log), axis=1)
        log_all_subjects = np.mean(np.array(log_all_channels), axis=0)

        # Compute bootstrapped confidence interval
        n_bootstrap = 1000  # Number of bootstrap samples
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(log_all_subjects, size=len(log_all_subjects), replace=True)
            bootstrap_mean = np.mean(bootstrap_sample)
            bootstrap_means.append(bootstrap_mean)
        bootstrap_means = np.array(bootstrap_means)
        lower_bound = np.percentile(bootstrap_means, 2.5)
        upper_bound = np.percentile(bootstrap_means, 97.5)

        log_both.append(log_all_subjects)

        high_sd = np.std(np.array(log_all_subjects), axis=0)
        
        plt.plot(log_freqs, log_all_subjects.T, label=labellist[index], color=colors[index], alpha=line_alpha)
        plt.legend()

        plt.fill_between(log_freqs, log_all_subjects.T - lower_bound, log_all_subjects.T + upper_bound, alpha=fill_alpha, color=colors[index], edgecolor=None)
    plt.title('Grand Average PSD', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    plt.ylabel('Power (mV$^2$/Hz)', fontsize=14, fontweight='bold')

    # Set x-axis tick labels to logarithmic form
    plt.xticks([np.log10(freqs[0]), np.log10(10), np.log10(freqs[-1])], [freqs[0], 10, freqs[-1]])

    # Set y-axis ticks to log scale
    # max_value = np.max(psda)
    # plt.yscale('log')
    plt.yticks([-2.25, -.75, .75, 2.25], ['', '', '10', '100'])

    plt.show(block=False)

plot_output(wake_powerspec, sleep_powerspec, ket_powerspec, prop_powerspec, xenon_powerspec, freqs)
#   os.chdir(save_dir)
plt.savefig("GA_psd_all_cond_1.svg", dpi=600, bbox_inches='tight')
plt.savefig("GA_psd_all_cond_1.png", dpi=600, bbox_inches='tight')
# %%
