import torch
import numpy as np
from matplotlib import pyplot as plt
import hht, visualization
from scipy.signal import chirp
import mne
from os.path import isfile, join
from os import listdir

def compute_average_timeseries(time_series):
    avg_t_s = []
    for i in range(len(time_series[0])):
        avg = 0
        for j in range(len(time_series)):
            avg += time_series[j][i]
        avg_t_s.append(avg / len(time_series))
    return np.array(avg_t_s)

file = r"C:\Users\svenm\OneDrive\Documenten\GitHub\BCI_project\eeg_reader\data\Alexander Jansson"
#data = mne.io.read_raw_edf(file, preload=True)

#

onlyfiles = [f for f in listdir(file) if isfile(join(file, f))]
print(onlyfiles)
eeg_training_data = []
low, hi = 0.2, 64
ideal_sample_rate = 128


new_ch_names = []

for a_file in onlyfiles:
    new_ch_names.append(a_file)
    eeg_training_data.append(mne.io.read_raw_edf(join(file, a_file)))


for a_file in eeg_training_data:
    a_file.load_data()


# creating a pretend signal so we can plot all PSD's (fourier components):
for signal in eeg_training_data:
    signal.filter(low, hi)
    signal.notch_filter(np.arange(60, 240, 60))
    signal.resample(ideal_sample_rate)


  
#new_info.__setitem__(key, val)
print("test")
print(eeg_training_data)

eeg_array = np.array([data.get_data()[0] for data in eeg_training_data])
new_info = mne.create_info(new_ch_names, ideal_sample_rate)
added_signals = mne.io.RawArray(eeg_array, new_info)
print(added_signals.info)
data = added_signals.get_data()

defaults = []
left_left = []
right_right = []

for i in range(len(data)):
    ch_name = added_signals.ch_names[i]

    if ch_name.startswith("Default"):
        defaults.append(data[i])

    if ch_name.startswith("Right"):
        right_right.append(data[i])

    if ch_name.startswith("Left"):
        left_left.append(data[i])

#
rr = compute_average_timeseries(right_right)

x = rr
fs = ideal_sample_rate



imfs, imfs_env, imfs_freq = hht.hilbert_huang(x, fs, num_imf=3)
visualization.plot_IMFs(x, imfs, fs)
spectrum, t, f = hht.hilbert_spectrum(imfs_env, imfs_freq, fs)
visualization.plot_HilbertSpectrum(spectrum, t, f)