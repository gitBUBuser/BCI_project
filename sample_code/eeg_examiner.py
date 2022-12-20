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
from scipy.signal import spectrogram
from scipy.signal import hilbert

path = "/home/baserad/Documents/Schoolwork/NDL/BCI_project/eeg_reader/EEG_Trainer/Data/Sven_training.edf"
processor = preprocessing.EEG_processer()


eeg_training_data = []
low, hi = 0.25, 45
ideal_sample_rate = 125


new_ch_names = []

raw_data = mne.io.read_raw_edf(path, preload=True)
raw_data.filter(low, hi)
raw_data.resample(ideal_sample_rate)


# Extract the annotations from the raw object
annotations = raw_data.annotations

# Create a matrix of events from the annotations

events = mne.events_from_annotations(raw_data)

event_ids = events[1]

epochs = mne.Epochs(raw_data, events[0], detrend=1)


fig, ax = plt.subplots(figsize=(20,5))
labels = []
fouriers = []
freq = []

def compute_FFT(signal):
    n = signal.size
    time = n / ideal_sample_rate
    ts = 1 / ideal_sample_rate

    #tl = np.arange(0, time, ts)
    FFT = abs(fft.fft(signal))[range(int(n/2))]
    fouriers.append(FFT)
    freq = np.array(fftpack.fftfreq(n, ts))[range(int(n/2))]

    return FFT, freq

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

FFTs = {}
freqs = []

epochs_by_annotations = {}

def add_to_epochs_by_annotation(key, value):

    if key in epochs_by_annotations.keys():
        epochs_by_annotations[key].append(value)
    else:
        epochs_by_annotations[key] = value

for key, value in event_ids.items():
    print(key)
    print(value)

    selection = epochs[value]
    for test in selection:
        print(test)
        
    print(selection)
    print(epochs)



for key, value in event_ids.items():

    selected_epochs = epochs.__getitem__(value)
    print(selected_epochs)
    add_to_epochs_by_annotation(key, selected_epochs)

    """inner_ffts = []
    for e in selected_epochs:
        ft, freqs = compute_FFT(e[0])
        inner_ffts.append(ft)

    FFTs[key] = compute_average(inner_ffts)"""

"""for key, value in FFTs.items():
    ax.plot(freqs, value, label=key)"""

for key, value in epochs_by_annotations.items():
    print(f"event: {key} || data: {value}")



    


"""


defaults_FFT = []
left_left_FFT = []
right_right_FFT = []

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

    if ch_name.startswith("default"):
        defaults_FFT.append(FFT)
        defaults.append(data[i])
        labels.append("Default")

    if ch_name.startswith("right"):
        right_right_FFT.append(FFT)
        right_right.append(data[i])
        labels.append("Right -> right")

    if ch_name.startswith("left"):
        left_left_FFT.append(FFT)
        left_left.append(data[i])
        labels.append("Left -> left")


print(right_right)
avg_rhr = compute_average(right_right_FFT)
avg_lhl = compute_average(left_left_FFT)
avg_def = compute_average(defaults_FFT)

#computes the average of the items in time_series
def compute_average_timeseries(time_series):
    avg_t_s = []
    for i in range(len(time_series[0])):
        avg = 0
        for j in range(len(time_series)):
            avg += time_series[j][i]
        avg_t_s.append(avg / len(time_series))
    return np.array(avg_t_s)
            


print(freq.shape)
print(avg_rhr.shape)

#hil_rr = hilbert(avg_rhr)

#ax.plot(freq, np.abs(hil_rr), label="hilbert_rr", c="orange")
ax.plot(freq, avg_def, label="default", c="red")
ax.plot(freq, avg_rhr, label="right->right", c="blue")
ax.plot(freq, avg_lhl, label="left->left", c="green")

plt.legend()
plt.show()

fig, ax = plt.subplots(3)

avg_ts_right = compute_average_timeseries(right_right)
avg_ts_left = compute_average_timeseries(left_left)
avg_ts_default = compute_average_timeseries(defaults)

f, t, Sxx = spectrogram(avg_ts_right, ideal_sample_rate)

ax[0].set_title("right -> right")
ax[0].pcolormesh(t, f, Sxx, shading="gouraud")
#ax[0].ylabel('Frequency [Hz]')
#ax[0].xlabel('Time [sec]')

f, t, Sxx = spectrogram(avg_ts_left, ideal_sample_rate)

ax[1].set_title("left -> left")
ax[1].pcolormesh(t, f, Sxx, shading="gouraud")
#ax[1].ylabel('Frequency [Hz]')
#ax[1].xlabel('Time [sec]')

f, t, Sxx = spectrogram(avg_ts_default, ideal_sample_rate)

ax[2].set_title("default")
ax[2].pcolormesh(t, f, Sxx, shading="gouraud")
#ax[2].ylabel('Frequency [Hz]')
#ax[2].xlabel('Time [sec]')


plt.show()
     

#labels = mne.label(["def1, def2, def3, lhl1, lhl2, lhl3, rhr1, rhr2, rhr3"])

#added_signals.get_channel.plot_psd(average=False, color="red", spatial_colors=False)
#added_signals[1].plot_psd(average=False, color="red", spatial_colors=False)



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