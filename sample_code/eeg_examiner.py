import numpy as np
import sklearn as sk
from sklearn import linear_model as lm
import mne as mne
import EEG_Preprocessing as preprocessing
import matplotlib.pyplot as plt
import scipy.signal as s_signal
from os import listdir
from scipy import fft
from scipy import fftpack
from os.path import isfile, join


path = "/home/baserad/Documents/Schoolwork/NDL/BCI_project/eeg_reader/data/Alexander Jansson/"
processor = preprocessing.EEG_processer()
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
print(onlyfiles)
eeg_training_data = []
low, hi = 0.2, 64
ideal_sample_rate = 128


new_ch_names = []

for a_file in onlyfiles:
    new_ch_names.append(a_file)
    eeg_training_data.append(mne.io.read_raw_edf(join(path, a_file)))


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

fig, ax = plt.subplots(figsize=(20,5))
labels = []
fouriers = []
freq = []

def compute_average(ffts):
    ffts_T = np.array(ffts).T
    average = []

    for i in range(len(ffts_T)):
        avg = 0
        for j in range(len(ffts_T[i])):
            avg += ffts_T[i][j]
        avg /= len(ffts_T[i])
        average.append(avg)

    return np.array(average)
        

defaults = []
left_left = []
right_right = []

for i in range(len(data)):
    signal = data[i]
    n = signal.size
    print(signal.size)
    time = n / ideal_sample_rate
    ts = 1 / ideal_sample_rate

    #tl = np.arange(0, time, ts)
    FFT = abs(fft.fft(data[i]))[range(int(n/2))]
    fouriers.append(FFT)
    freq = np.array(fftpack.fftfreq(n, ts))[range(int(n/2))]
    ch_name = added_signals.ch_names[i]

    if ch_name.startswith("Default"):
        defaults.append(FFT)
        labels.append("Default")

    if ch_name.startswith("Right"):
        right_right.append(FFT)
        labels.append("Right -> right")

    if ch_name.startswith("Left"):
        left_left.append(FFT)
        labels.append("Left -> left")


print(right_right)
avg_rhr = compute_average(right_right)
avg_lhl = compute_average(left_left)
avg_def = compute_average(defaults)
added_signals.plot()
plt.show()
"""default_signal = added_signals.get_data()[0]
fr = 128
samp = len(default_signal)
time = samp / fr
ts = 1 / fr

time_line = np.arange(0, time, ts)

hilbert = s_signal.hilbert(default_signal)
analytic_signal = abs(hilbert)
ax.plot(time_line, analytic_signal, label="hilbert", c="yellow")
ax.plot(time_line, default_signal, label="default", c="blue")
plt.legend()
plt.show()"""

print(freq.shape)
print(avg_rhr.shape)

ax.plot(freq, avg_def, label="default", c="red")
ax.plot(freq, avg_rhr, label="right->right", c="blue")
ax.plot(freq, avg_lhl, label="left->left", c="green")

plt.legend()



plt.show()



     
#labels = mne.label(["def1, def2, def3, lhl1, lhl2, lhl3, rhr1, rhr2, rhr3"])

#added_signals.get_channel.plot_psd(average=False, color="red", spatial_colors=False)
#added_signals[1].plot_psd(average=False, color="red", spatial_colors=False)


"""
# preprocessing stuff with the signal

signal.filter(low, hi)
signal.notch_filter(np.arange(60, 240, 60))
signal.resample(128)
sig = s_signal.detrend(signal.get_data()[0])
# getting info from the signal (for plotting)
sig_info = signal.info
print(sig_info)
o_sfreq = sig_info["sfreq"]
samples = signal.__len__()
seconds = samples / o_sfreq
ts = 1 / o_sfreq
tl = np.arange(0, seconds, ts)

signal.plot_psd()


fig, ax = plt.subplots(figsize = (30,3))

# doing hilbert transform
hilbert = s_signal.hilbert(sig)
analytic_signal = abs(hilbert)


ax.plot(tl, sig, label = "signal")
ax.plot(tl, analytic_signal, label = "hilbert")
fig.legend()
plt.show()


signal.filter(low, hi)
signal.resample(128)


signal.plot()

plt.show()
"""