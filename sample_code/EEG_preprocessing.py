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
import sklearn as sk
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import resample
from scipy.signal import butter
from scipy.signal import sosfilt

path = "/home/baserad/Documents/Schoolwork/NDL/BCI_project/eeg_reader/EEG_Trainer/Min_lilla_hjärna.edf"
low, hi = 2, 30

class BlankObj:
  def __repr__(self):
   return ""

class DataHandler():
    def __init__(self, filter_band, sampling_rate):
        self.ts_data = []
        self.label_names = []

        self.filter_band = filter_band
        self.sampling_rate = sampling_rate

        self.sos = butter(4, self.filter_band, btype="bandpass", output="sos", fs=10000)

    def load_file(self, path):

        raw = mne.io.read_raw_edf(path, preload=True)
        raw.filter(self.filter_band[0], self.filter_band[1])
        raw.resample(self.sampling_rate)

        event_ids, epochs = self.raw_to_epochs(raw)
        self.epochs_to_signals(epochs, event_ids)
        print(self.ts_data)

    def load_array(self, X, y = None):
        X = np.array(X)
        if y != None:
            self.ts_data.append([X, y])
        else:
            self.ts_data.append([X, BlankObj()])

    def filter_and_resample(self, X):
        X = np.array(X)
        filtered_X = sosfilt(self.sos, x)
        return resample(filtered_X, self.sampling_rate)

    def set_signal(self, X, y = None):
        X = np.array(X)
        if y == None:
            self.ts_data = [[X], BlankObj()]
        else:
            self.ts_data = [[X], y]

    def load_and_filter_array(self, X, y = None):
        X = np.array(X)
        filtered_X = sosfilt(self.sos, X)
        X = resample(filtered_X, self.sampling_rate)
        if y != None:
            self.ts_data.append([filtered_X, y])
        else:
            self.ts_data.append([filtered_X, BlankObj()])

    def clear_data(self):
        self.ts_data.clear()
    
    def classes(self):
        return [signal[1] for signal in self.ts_data]

    def epochs_to_signals(self, epochs, event_ids):
        for name, id in event_ids.items():
            class_id = id - 1
            self.label_names.append(name)
            signals = epochs.__getitem__(str(id))
            for signal in signals:
                self.load_array(signal[0], class_id)

    def no_outlier_indeces(self, data, m = 4.):
        index_array = np.arange(0, len(data), 1)
        median = np.median(data)
        MAD = np.median([abs(data_point-median) for data_point in data])
        return index_array[abs(data - median) < m * MAD]

    def raw_to_epochs(self, raw):
        durations = raw.annotations.duration
        mean_duration = np.mean(durations)
        ham_indeces = self.no_outlier_indeces(durations)
        events, event_ids = mne.events_from_annotations(raw, verbose=True)
        events = events[ham_indeces]
        return event_ids, mne.Epochs(raw, events, preload=True, tmin=0, tmax=mean_duration, baseline=None)

    def get_FFTs(self):
        FFTs = []
        freqs = []
        labels = []

        for data in self.ts_data:
            X = data[0]
            labels.append(data[1])

            n = X.size
            time = n / self.sampling_rate
            ts = 1 / self.sampling_rate

            FFT = abs(fft.fft(signal))[range(int(n/2))]
            freq = np.array(fftpack.fftfreq(n, ts))[range(int(n/2))]
            FFTs.append(FFT)
            freqs.append(freq)
        
        return FFTs, freqs, labels
    
    def compute_average_timeseries(self, id = None, interval = 1):
        signals = np.array(self.ts_data)
        t_signals = signals.T
        label_count = len(self.label_names)
        new_labels = []
        mean_signals = []
        for i in range(label_count):
            
            labeled_signals = np.array([signal[0] for signal in signals if signal[1] == i])
            label = i
            step = interval
            avg_signals = {}

            for z in range(0, labeled_signals.shape[0], step):
                avg_signals[z] = []

            for a in range(0, labeled_signals.shape[1]):
                for b in range(0, labeled_signals.shape[0], step):
                    index = b
                    avg = 0
                    if b + step <= labeled_signals.shape[0]:
                        for c in range(b, b + step):
                            avg += labeled_signals[c][i]
                        avg_signals[index].append(avg / step)
                    else:
                        pass
            

            for index, value in avg_signals.items():
                if value != []:
                    mean_signals.append(value)
                    new_labels.append(i)

        self.clear_data()
        for y in range(len(mean_signals)):
            self.load_array(mean_signals[y], new_labels[y])

    def average_series(self, series, labels = None, interval = 1):
        series = np.array(series)

        new_labels = []
        mean_series = []
        
        if labels != None:
            label_count = len(list(dict.fromkeys(labels)))
            for i in range(label_count):
                avg_series = {}

                for z in range(0, series.shape[0], interval):
                        avg_series[z] = []

                for a in range(0, series.shape[1]):
                    for b in range(0, series.shape[0], interval):
                        index = b
                        avg = 0
                        if b + interval <= series.shape[0]:
                            for c in range(b, b + interval):
                                avg += series[c][i]
                            avg_series[index].append(avg / interval)
                        else:
                            pass
            
                for index, value in avg_series.items():
                    if value != []:
                        mean_series.append(value)
                        new_labels.append(i)
        else:
            pass

        return mean_series, new_labels

    def get_bands(self, low, hi, amount):
        difference = hi - low
        step = difference / amount
        bands = []
        for i in np.arange(low, hi, step):
            bands.append((i, i+step))
        return bands

    def bandpowers(self, bands):
        bp_signals = []
        labels = []

        for signal in self.ts_data:
            signal_bps = []
            labels.append(signal[1])
            for band in bands:
                signal_bps.append(self.bandpower(signal[0], self.sampling_rate, band)) 
            bp_signals.append(signal_bps)

        return bp_signals, labels       


    def bandpower(self, signal, sf, band):
        band = np.asarray(band)
        low, hi= band

        def get_bandpower(signal):
            f, Pxx = scipy.signal.periodogram(signal, fs=sf)
            ind_min = np.argmax(f > low) - 1
            ind_max = np.argmax(f > hi) - 1
            return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

        bps = get_bandpower(signal)
        return bps

    def std_per_band(self, epoch_dict):
        std_bands = {}
        for key in epoch_dict.keys():
            std_bands[key] = np.std(epoch_dict[key])

    def segment(self, time_per_segment):
        segmented_signals = []
        labels = []
        for signal in self.ts_data:
            y = signal[1]
            X = signal[0]
            samples = len(X)
            samples_per_segment = time_per_segment * self.sampling_rate
            for i in np.arange(0, samples, samples_per_segment):
                if int(i + samples_per_segment) <= samples:
                    segmented_signals.append(X[int(i):int(i + samples_per_segment)])
                    labels.append(y)
        self.clear_data()

        for j in range(len(segmented_signals)):
            if segmented_signals[j] != []:
                self.load_array(segmented_signals[j], labels[j])
            else:
                print("error")

            


    def get_best_filter_bands(self, cov1, cov2, amount = 1, every_other = False):
        average_cov = cov1 + cov2
        D, V = la.eig(cov1, average_cov)

        D_sort = np.argsort(D)[::-1]
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

    def __str__(self):
        return f"Labels: {self.label_names}\nBand: {self.filter_band}\nSampling rate: {self.sampling_rate}\nClasses: {self.classes()}\nSignals: {np.array(self.ts_data, dtype=object).shape[0]}"

    def print_signals(self):
        for signal in self.ts_data:
            print(f"Label: '{signal[1]}'   |   > {signal[0]}")




