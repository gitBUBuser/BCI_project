import numpy as np 
from scipy.io import wavfile
import sklearn as sk
import mne
import scipy.signal as signal


class EEG_processer:
    def __init__(self):
        self.filter_band = (1.5, 16)
        self.inteval = (-0.1, 0)


    def bandpass_filter_MNE(self, some_data):
        return some_data.filter(self.filter_band[0], self.filter_band[1], method="icir")

    def preprocessed_to_epoch(self, preprocessed_data, decimate=10, baseline_ival=(-.2, 0)):
        class_ids = { "Left": 1, "Right": 2, "StartShoot:" 3, "StopShoot": 4}

        events = mne.events_from_annotations(preprocessed_data, event_id=class_ids)[0]
        epo_data = mne.Epochs(preprocessed_data, events, event_id=class_ids,
                              baseline=baseline_ival, decim=decimate,
                              reject=reject, proj=False, preload=True)
        return epo_data

    def correct_for_drift(self, some_data):
        some_data.apply_baseline(self.inteval)



    






    

    def get_sampling_rate(self):
        return self.sample_rate
        
    def number_of_samples(self, some_data):
        return len(some_data[0])
    
    def get_seconds(self, some_data, some_sample_rate):
        return self.number_of_samples(some_data) / float(some_sample_rate)
    
    def get_time_step(self, some_sample_rate):
        return 1 / some_sample_rate

    def load_data(self, data_path):
        return wavfile.read(data_path)
    

    def remove_outliers(self):
        pass
    
    def epoch_data(self, some_data, time_step):
        pass

    def bandpass_filter(self):
        b, a = self.butter_bandpass()
        self.data = signal.lfilter(b, a, self.data)

    def butter_bandpass(self, order = 5):
        nyquist = 0.5 * self.get_sampling_rate()
        low = self.filter_band[0] / nyquist
        high = self.filter_band[1] / nyquist
        return signal.butter(order, [low, high], btype = "band")

