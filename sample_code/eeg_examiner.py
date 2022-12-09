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

path = "/home/baserad/Documents/Schoolwork/NDL/BCI_project/eeg_reader/EEG_Trainer/Data"
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