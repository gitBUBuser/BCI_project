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
import scipy as sc
import random

path = "/home/baserad/Documents/Schoolwork/NDL/BCI_project/eeg_reader/EEG_Trainer/Min_lilla_hjärna.edf"
low, hi = 2, 25
ideal_sample_rate = 100

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def compute_FFT(signal):
    n = signal.size
    time = n / ideal_sample_rate
    ts = 1 / ideal_sample_rate

    #tl = np.arange(0, time, ts)
    FFT = abs(fft.fft(signal))[range(int(n/2))]
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

def compute_average_timeseries(time_series):
    avg_t_s = []
    t_series = np.array(time_series)
    t_series_shape = t_series.shape

    for i in range(t_series_shape[1]):
        avg = 0
        for j in range(t_series_shape[0]):
            avg += time_series[j][i]
        avg_t_s.append(avg)
    
    return np.array(avg_t_s)
            

eeg_training_data = []

new_ch_names = []

raw_data = mne.io.read_raw_edf(path, preload=True)

sfreq = raw_data.info["sfreq"]
raw_data.filter(low, hi)
raw_data.resample(ideal_sample_rate)

annotations = raw_data.annotations

durations = annotations.duration

def average_power_across_band(FFTs, freqs, interval):

    new_freqs = []
    new_FFTs = []
    for i in range(0, len(freqs), interval):
        avg_FFT = 0
        avg_freqs = 0
        try:
            for j in range(0, interval):
                index = i + j
                avg_FFT += FFTs[index]
                avg_freqs += freqs[index]

            avg_FFT /= interval
            avg_freqs /= interval
            new_FFTs.append(avg_FFT)
            new_freqs.append(avg_freqs)
        except:
            continue
    
    return new_FFTs, new_freqs


def no_outlier_indeces(data, m = 4.):
    index_array = np.arange(0, len(data), 1)

    median = np.median(data)
    MAD = np.median([abs(data_point-median) for data_point in data])
    print()
    print(MAD)
    print(median)
    print(m*MAD)
    print(abs(data - median))
    print()
    return index_array[abs(data - median) < m * MAD]

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def remove_outlier_epochs_given_label(epoch_dict, m = 50):
    kept_signals = {}
    for key in epoch_dict.keys():
        signals = epoch_dict[key]
        kept_signals[key] = np.arange(0, len(signals), 1)
        inner_index_arrays = np.ones(len(signals))
        for i in range(len(signals)):
            median = np.median(signals[i])
            MAD = np.median([abs(data_point-median) for data_point in signals[i]])
            for j in range(len(signals[i])):
                if signals[i][j] - median > m * MAD:
                    inner_index_arrays[i] = 0
                    break
           

        print(inner_index_arrays)
        bolean_arr = np.array([True if ele == 1 else False for ele in inner_index_arrays])
        kept_signals[key] = kept_signals[key][bolean_arr]
    return kept_signals

ham_indeces = no_outlier_indeces(durations)
removed_outlier_durations = durations[ham_indeces]
mean_duration = np.mean(removed_outlier_durations)


#   raw_data.filter(low, hi)


# Create a matrix of events from the annotations

events, event_ids = mne.events_from_annotations(raw_data, verbose=True)
events = events[ham_indeces]

print(events)
print(event_ids)

epochs = mne.Epochs(raw_data, events, preload=True, tmax=mean_duration)
print(epochs)
epochs.drop_bad()

epoch_dict = {}
signals_dict = {}
FFTs = {}
freqs = []


for key, value in event_ids.items():
    signals = epochs.__getitem__(str(value))
    inner_ffts = []
    inner_signals = []

    for sig in signals:
        N = len(sig[0])
        T = 1 / ideal_sample_rate
        x = np.linspace(0.0, N*T, N)
        Y = sc.fftpack.fft(sig[0])
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        freqs = xf

        inner_ffts.append(2.0 /N * np.abs(Y[:N//2]))
        inner_signals.append(sig[0])
    signals_dict[key] = np.array(inner_signals)
    FFTs[key] = np.array(inner_ffts)

print(freqs)
#non_drops = remove_outlier_epochs_given_label(signals_dict)
#print(non_drops["default"])


"""
for key in signals_dict.keys():
    signals_dict[key] = signals_dict[key][non_drops[key]]
for key in FFTs.keys():
    FFTs[key] = FFTs[key][non_drops[key]]

plausible_bands = []
for i in range(3000):
    p_low = random.randrange(2, 40)
    p_hi = random.randrange(p_low + 3, p_low + 12)
    plausible_bands.append((p_low,p_hi))
    
highest_bands_for_keys = []
highest_power = []
for key, value in signals_dict.items():
    max_band = 0
    best_band = None
    avg = compute_average_timeseries(value)
    analytic_avg = hilbert(avg)
    amplitude_envelope = np.abs(analytic_avg)

    time = len(avg) / ideal_sample_rate
    ts = 1 / ideal_sample_rate
    time = np.arange(0, time, ts)
    
    print("________________________________________________________________________")
    print()
    for band in plausible_bands:
        power = bandpower(np.abs(avg), sf= ideal_sample_rate, band=band)
        print(f"CONDITION: {key} || BAND: {band} << POWER: {power}")
        if power > max_band:
            max_band = power
            best_band = band
    highest_bands_for_keys.append(best_band)
    highest_power.append(max_band)

for i in range(len(highest_bands_for_keys)):
    print("HIGHBAND: " + str(highest_bands_for_keys[i]))
    print("POWER: " + str(highest_power[i]))

    plt.plot(time,amplitude_envelope, label = key)
"""
cmap = get_cmap(len(FFTs.keys()) + 1)
indexu = 0

fig, ax = plt.subplots(sharex=True, sharey=True)
ax.set_facecolor("black")
for key, values in FFTs.items():

    c = cmap(indexu)
    indexu += 1
    avg_ffts = compute_average(np.abs(values))
    interval = 5
    
    
    avg_across_band, new_freqs = average_power_across_band(avg_ffts, freqs, interval)
    if key == "default" or key == "think of being happy" or key == "right hand right":
        ax.plot(new_freqs, np.log(np.abs(avg_across_band)), label = key + " avg", color = c)
    ax.scatter(new_freqs, np.log(np.abs(avg_across_band)), label = key + " avg", color = c)
    ax.plot(freqs, np.log(np.abs(avg_ffts)), label = key, alpha = 0.5, color = c)
  #  ax.scatter(indexu, 1)

plt.show()
plt.legend()
fig, ax = plt.subplots(sharex=True, sharey=True)
ax.set_facecolor("black")
cmap = get_cmap(len(signals_dict.keys()) + 1)
indexu = 0

for key, values in signals_dict.items():
    c = cmap(indexu)
    indexu += 1
    avg = compute_average_timeseries(values)
    print(avg)

    x_t = np.arange(0, len(avg) / ideal_sample_rate, 1/ ideal_sample_rate)
    hil = hilbert(avg)
    amplitude_envelope = np.abs(hil)
    interval = 20
    avg_envelope = []
    avg_x = []
    for x in range(0,len(amplitude_envelope), interval):
        avg = 0
        avg_2 = 0
        try:
            for i in range(x, x + interval):
                avg += amplitude_envelope[i]
                avg_2 += x_t[i]
            avg_2 /= interval
            avg /= interval

            avg_x.append(avg_2)
            avg_envelope.append(avg)
        except:
            continue
    


    ax.plot(x_t, np.log(amplitude_envelope), label = key, color = c, alpha = 0.3)
    ax.plot(avg_x, np.log(avg_envelope), label = key + " envelope", color = c)

plt.legend()
plt.show()

"""

# sample spacing
    T = 1.0 / ideal_sample_rate
    Y    = numpy.fft.fft(y)
    freq = numpy.fft.fftfreq(len(y), t[1] - t[0])
    x = np.linspace(0.0, N*T, N)
    y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.show()
 
for key in epochs_by_annotations.keys():
    print(epochs_by_annotations[key].get_data())
    print(len(epochs_by_annotations[key].get_data()))
    average_signals[key] = epochs_by_annotations[key].average()

print("shit")
for key in average_signals:
    print(key)
    data = average_signals[key].get_data()
    print(len(data[0]))

    time_step = len(average_signals) / ideal_sample_rate
    ts = np.arange(0, len(average_signals), time_step)
    ax.plot(ts,data, label=key)

fig, ax = plt.subplots(figsize=(20,5))
labels = []
fouriers = []
freq = []

plt.legend()
plt.show()


for key, value in event_ids.items():

    selected_epochs = epochs.__getitem__(value)
    print(selected_epochs)
    add_to_epochs_by_annotation(key, selected_epochs)

    inner_ffts = []
    for e in selected_epochs:
        ft, freqs = compute_FFT(e[0])
        inner_ffts.append(ft)

    FFTs[key] = compute_average(inner_ffts)

for key, value in FFTs.items():
    ax.plot(freqs, value, label=key)

for key, value in epochs_by_annotations.items():
    print(f"event: {key} || data: {value}")
"""
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