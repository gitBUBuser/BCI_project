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
import random
from scipy.signal import detrend

"""
    This file includes the central features we use to preprocess EEG signals before they can be used for classification.
    The preprocesser works for both labaled and unlabaled data,
    both raw numpy arrays and edf files.
    Epoched data and non epoched data.
"""

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

    def filter_and_resample(self, X, seconds):
        X = np.array(X)
        filtered_X = sosfilt(self.sos, X)
        return resample(filtered_X, self.sampling_rate * seconds)

    def set_signal(self, X, y = None):
        X = np.array(X)
        if y == None:
            self.ts_data = [[X, BlankObj()]]
        else:
            self.ts_data = [[X, y]]

    def overwrite_signal(self, data):
        self.ts_data = data

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

    def detrend(self):
        new_data = []
        for signal in self.ts_data:
            labaled_signal = signal[0]
            detrended_signal = detrend(labaled_signal)
            new_data.append([detrended_signal, signal[1]])
        self.overwrite_signal(new_data)

        



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

    def average_series(self, series, labels = None, interval = 1, shuffle = False):
        if labels != None:
            signals_by_label = []
            averaged_signals = []
            label_count = len(list(dict.fromkeys(labels)))
    
            for a in range(label_count):
                signals_by_label.append([])
                labels_indexes = [index for index in range(0, len(labels)) if labels[index] == a]
                for index in labels_indexes:
                    signals_by_label[a].append(series[index])

            if shuffle:
                for i in range(len(signals_by_label)):
                    random.shuffle(signals_by_label[i])
            for z in range(len(signals_by_label)):
                signals = signals_by_label[z]
                t_steps = int(len(signals) / interval)
                avg_series = {}

                for indexus in range(t_steps):
                    avg_series[indexus] = []
                
                for k in range(len(signals[0])):
            
                    index = 0
                    for j in range(0, len(signals), interval):
                        avg = 0
                        if j + interval <= len(signals):
                            for c in range(j, j + interval):
                                avg += signals[c][k]
                            avg_series[index].append(avg / interval)
                            index += 1
                        else:
                            continue

                for index, serie in avg_series.items():   
                    averaged_signals.append([serie, z])

            #print(averaged_signals)
            return [signal[0] for signal in averaged_signals], [signal[1] for signal in averaged_signals]
        else:
            averaged_signals = []
            signals = series
            t_steps = int(len(signals) / interval)
            avg_series = {}

            for indexus in range(t_steps):
                avg_series[indexus] = []

            for k in range(len(signals[0])):
            
                index = 0
                for j in range(0, len(signals), interval):
                    avg = 0
                    if j + interval <= len(signals):
                        for c in range(j, j + interval):
                            avg += signals[c][k]
                        avg_series[index].append(avg / interval)
                        index += 1
                    else:
                        continue

            for index, serie in avg_series.items():   
                averaged_signals.append([serie, BlankObj()])
            return [signal[0] for signal in averaged_signals], [signal[1] for signal in averaged_signals]
            
            

    
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
        signal = np.array(signal)
        band = np.asarray(band)
        low, hi= band
        def get_bandpower(signal):
            f, Pxx = scipy.signal.periodogram(signal, fs=sf)
            ind_min = np.argmax(f > low) - 1
            ind_max = np.argmax(f > hi) - 1
            return np.log(np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max]))

        bps = get_bandpower(signal)
        return bps

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

    def __str__(self):
        return f"Labels: {self.label_names}\nBand: {self.filter_band}\nSampling rate: {self.sampling_rate}\nClasses: {self.classes()}\nSignals: {np.array(self.ts_data, dtype=object).shape[0]}"

    def print_signals(self):
        for signal in self.ts_data:
            print(f"Label: '{signal[1]}'   |   > {signal[0]}")
