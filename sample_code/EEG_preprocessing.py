from scipy.signal import welch
from scipy.integrate import simps
import mne
import numpy as np
import random
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import operator
import scipy.linalg as la

path = "/home/baserad/Documents/Schoolwork/NDL/BCI_project/eeg_reader/EEG_Trainer/Min_lilla_hjärna.edf"
low, hi = 2, 30
ideal_sample_rate = 100

class Preprocessor():
    def __init__(self):
        pass
    
    def get_event_ids_from_raw_edf(self, data):
        return mne.events_from_annotations(raw_data, verbose=True)[1]

    def get_epoch_dict_from_edf_path(self, path, sample_rate, low, hi):
        raw_data = mne.io.read_raw_edf(path, preload=True)

        raw_data.filter(low, hi)

        sfreq = raw_data.info["sfreq"]

        if sfreq != sample_rate:
            raw_data.resample(ideal_sample_rate)
        
        events, event_ids, epochs = self.raw_edf_to_epochs(raw_data)
        return self.epochs_to_dict(epochs, event_ids)


    def raw_edf_to_epochs(self, raw):
        def no_outlier_indeces(data, m = 4.):
            index_array = np.arange(0, len(data), 1)

            median = np.median(data)
            MAD = np.median([abs(data_point-median) for data_point in data])
            return index_array[abs(data - median) < m * MAD]

       
        durations = raw.annotations.duration
        mean_duration = np.mean(durations)

        ham_indeces = no_outlier_indeces(durations)

        events, event_ids = mne.events_from_annotations(raw, verbose=True)
        events = events[ham_indeces]
        return events, event_ids, mne.Epochs(raw, events, preload=True, tmax=mean_duration)

    def compute_FFT(self, signal):
        n = signal.size
        time = n / ideal_sample_rate
        ts = 1 / ideal_sample_rate

        #tl = np.arange(0, time, ts)
        FFT = abs(fft.fft(signal))[range(int(n/2))]
        freq = np.array(fftpack.fftfreq(n, ts))[range(int(n/2))]

        return FFT, freq

    def epochs_to_dict(self, epochs, event_ids):
        epoch_dict = {}
        for key, value in event_ids.items():
            signals = epochs.__getitem__(str(value))
            epoch_dict[key] = signals
        return epoch_dict
    
    def dict_to_labeled_data(self):
        pass

    def compute_average_timeseries(self, time_series):
        avg_t_s = []
        t_series = np.array(time_series)
        t_series_shape = t_series.shape

        for i in range(t_series_shape[1]):
            avg = 0
            for j in range(t_series_shape[0]):
                avg += time_series[j][i]
            avg_t_s.append(avg)
    
        return np.array(avg_t_s)

    def ffts_from_dict(self, epoch_dict):
        fft_dict = {}
        for key in epoch_dict.keys():
            signals = epoch_dict[key]
            fft_signals = []
            for signal in signals:
                for channel_signal in signal:
                    fft_signals.append(self.compute_FFT(channel_signal))
            fft_dict[key] = fft_signals
        return fft_dict

    def average_dict_series(self, epoch_dict, average_index):
        new_dict = {}
        for key in epoch_dict.keys():
            signals = epoch_dict[key]
            new_signals = []
            for i in range(0, len(signals), average_index):
                inner_signals = []
                for j in range(i, i + average_index):
                    if j < len(signals):
                        inner_signals.append(signals[j])
                    else: 
                        continue
   
                new_signals.append(self.compute_average_timeseries(inner_signals))
            new_dict[key] = np.array(new_signals)
        return new_dict


        
    
    def bandpower(self, signal, sf, band):
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
        band = np.asarray(band)
        low, hi= band

        def get_bandpower(signal):
            f, Pxx = scipy.signal.periodogram(signal, fs=sf)
            ind_min = np.argmax(f > low) - 1
            ind_max = np.argmax(f > hi) - 1
            return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

        bps = []
        for ch in signal:
            bps.append(get_bandpower(ch))
        return bps

        def std_per_band(self, epoch_dict):
            std_bands = {}
            for key in epoch_dict.keys():
                std_bands[key] = np.std(epoch_dict[key])

    def get_best_filter_bands(self, cov1, cov2, amount = 1, every_other = False):
        average_cov = cov1 + cov2
        D, V = la.eig(cov1, average_cov)

        D_sort = np.argsort(D)
        D = D[D_sort]
        V = V[D_sort]

        if not every_other:
            return(D[:amount], V[:amount])
        else:
            new_d = []
            new_v = []

            index = 0
            positive_index = 0
            negative_index = -1
            positive_turn = True
            while index < amount:
                index += 1

                if positive_turn:
                    positive_turn = False
                    new_d.append(D[positive_index])
                    new_v.append(V[positive_index])
                    positive_index += 1
                else:
                    positive_turn = True
                    new_d.append(D[negative_index])
                    new_v.append(V[negative_index])
                    negative_index -= 1

            return (new_d, new_v)
                

if __name__ == "__main__":
    preprocessor = Preprocessor()
    epoch_dict = preprocessor.get_epoch_dict_from_edf_path(path, 240, 0.5, 40)

    frequency_bands = 10

    center_function = lambda x: x - x.mean()
    test_bands = []
    random.seed()  
    lowest_frequency = 2
    highest_frequency = 35
    difference = highest_frequency - lowest_frequency
    add_amount = difference / frequency_bands

    highs = np.arange(lowest_frequency + add_amount, highest_frequency + add_amount, add_amount)
    lows = np.arange(lowest_frequency, highest_frequency, add_amount)

    for i in range(len(highs)):
        test_bands.append((lows[i], highs[i]))

        

    for key in epoch_dict.keys():
        new_values = []
        for value in epoch_dict[key]:
            inner_band_values = []
            for band in test_bands:
                inner_band_values.append(np.log(preprocessor.bandpower(value, 240, band)))
            new_values.append(inner_band_values)
        epoch_dict[key] = np.array(new_values)

    epoch_dict = preprocessor.average_dict_series(epoch_dict, )

    happy = epoch_dict["think of being happy"]
    happy = np.array(happy.T[0].T)
    right = epoch_dict["right hand right"]
    right = np.array(right.T[0].T)

    happy_t = happy.T
    right_t = right.T
    happy_covariance = np.cov(center_function(happy_t))
    right_covariance = np.cov(center_function(right_t))

    d, v = preprocessor.get_best_filter_bands(happy_covariance, right_covariance, amount=10, every_other=True)
    print(d)
    right_in_dir = []
    happy_in_dir = []
    print(happy.shape)


    avg_happy = preprocessor.compute_average_timeseries(happy)
    avg_right = preprocessor.compute_average_timeseries(right)
    
    plt.plot(avg_right, label = "right")
    plt.plot(avg_happy, label = "happy")
    plt.title("average bandpower of 11 trials")
    plt.legend()
    plt.show()

    new_happy_vals = []
    new_x = []
    new_right_vals = []



    for i in range(len(d)):
        avg_happy_new = np.dot(v[i], happy_t)
        avg_right_new = np.dot(v[i], right_t)
        new_x.append(np.ones(len(avg_right_new)) * i)
        difference = np.abs(np.var(avg_happy_new)) - np.abs(np.var(avg_right_new))
        other_way = np.abs(np.var(avg_right_new)) - np.abs(np.var(avg_happy_new))
        print(str(difference) + "       " + str(other_way))
        new_happy_vals.append(avg_happy_new)
        new_right_vals.append(avg_right_new)

    plt.axhline(y = 0, color = 'r', linestyle = '--', label="proposed_decision_boundary")
    plt.title("'power' of trial projected onto eigenvector of frequencybands with most variation")
    plt.scatter(new_x, new_happy_vals, label = "happy")
    plt.scatter(new_x, new_right_vals, label = "right")
    plt.legend()
    plt.show()
    
    plt.scatter(new_happy_vals[2], new_happy_vals[1])
    plt.scatter(new_right_vals[2], new_right_vals[1])
    plt.show()

    """for i in range(len(d)):
        avg_happy_new = np.dot(v[i], happy_t)
        avg_right_new = np.dot(v[i], right_t)
        right_tl = np.ones(len(avg_right_new)) * i
        happy_tl = np.ones(len(avg_happy_new)) * i

        plt.scatter(right_tl, avg_right_new, label = "right")
        plt.scatter(happy_tl, avg_happy_new, label = "happy")

    plt.title("average bandpower of 11 trials")
    plt.legend()
    plt.show()

    """

    """
    for i in range(len(d)):



    for i in range(len(d)):
        print(v[i])
        happy_in_dir.append(np.dot(happy, v[i]))
        right_in_dir.append(np.dot(right, v[i]))
    

    plt.scatter(happy_in_dir[0], happy_in_dir[1], label = "happy")
    plt.scatter(right_in_dir[0],  right_in_dir[1], label = "right")
    plt.legend()
    plt.show()
"""
"""
    std_per_band_happy = {}
    std_per_band_right = {}

    for i in range(len(test_bands)):
        std_per_band_happy[str(test_bands[i])] = np.std(happy_t[i])
        std_per_band_right[str(test_bands[i])] = np.std(right_t[i])
    
    sorted_band_right = dict(sorted(std_per_band_right.items(), key=operator.itemgetter(1)))
    sorted_band_happy = dict(sorted(std_per_band_happy.items(), key=operator.itemgetter(1)))



    fig, ax = plt.subplots(ncols = 2, sharey=True, figsize=(19, 15))
    ax[0].bar(np.arange(len(sorted_band_happy)), sorted_band_happy.values(), label=sorted_band_happy.keys())
    ax[0].title.set_text("STDs think of being happy")
    ax[0].set_ylabel("STD [µV]")
    ax[0].set_xlabel("bands")
    ax[0].set_xticklabels(ax[0].get_xticks(), rotation = 60)
    ax[0].set_xticklabels(sorted_band_happy.keys())

    ax[1].bar(np.arange(len(sorted_band_right)), sorted_band_right.values(), label=sorted_band_right.keys())
    ax[1].set_ylabel("STD [µV]")
    ax[1].title.set_text("STDs think of moving hand to the right")
    ax[1].set_xlabel("bands")
    ax[1].set_xticklabels(ax[1].get_xticks(), rotation = 60)
    ax[1].set_xticklabels(sorted_band_right.keys())

    plt.show()

    



    happy_frame = pd.DataFrame(happy)
    right_frame = pd.DataFrame(right)

    fig, ax = plt.subplots(ncols = 2, figsize=(19, 15), sharey= True)   

    ax[0].title.set_text("correlation think of being happy")
    ax[0].matshow(happy_frame.corr())
    ax[0].set_xticklabels([''] + test_bands)
    ax[0].set_yticklabels([''] + test_bands)

    ax[1].title.set_text("correlation right hand right")
    mat = ax[1].matshow(right_frame.corr())
    ax[1].set_xticklabels([''] + test_bands)
    ax[1].set_yticklabels([''] + test_bands)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    fig.colorbar(mat, cax=cbar_ax)
   
   
    plt.show()"""
    #cov_happy = np.cov(happy)
    #cov_right = np.cov(right)

    #print(cov_happy)
    #covarience = np.cov()






    







