import numpy as np
import sklearn as sk
from sklearn import linear_model as lm
import mne as mne
import EEG_Preprocessing as preprocessing
import matplotlib as plt

path = "/home/baserad/Documents/BYB_Recording_2022-10-14_10.12.46_edited_edited.edf"
eeg_training_data = mne.io.read_raw_edf(path)
eeg_training_data.load_data()
processor = preprocessing.EEG_processer()



eeg_training_data = processor.bandpass_filter_MNE(eeg_training_data)
event_ids = {"blink": 1, "non-target": 0}

events = mne.events_from_annotations(eeg_training_data, event_ids)[0]
print(events)

epoched_data = mne.Epochs(eeg_training_data, events, event_id=event_ids,decim=1,proj=False,preload=True)
evo_t = epoched_data["blink"]
evo_nt = epoched_data["non-target"]

blinks = []
others = []

#Transform epoch into feature vector
samples_n = 10
training_samples = []

for epoch in evo_t:
    training_samples.append(1, sum(epoch.data) / 7001)
    """sample = []
    step = int(epoch.shape[1] / samples_n)
    for i in range(0,epoch.shape[1], step):
        if len(sample) == samples_n:
            continue
        else:
            sample.append(epoch[0][i])

    if len(sample) == samples_n:
        training_samples.append(np.concatenate([[1], sample]))"""

for epoch in evo_nt:
    training_samples.append(0, sum(epoch.data) / 7001)
    """sample = []
    step = int(epoch.shape[1] / samples_n)
    for i in range(0, epoch.shape[1], step):
        if len(sample) == samples_n:
            continue
        else:
            sample.append(epoch[0][i])
    if len(sample) == samples_n:
        training_samples.append(np.concatenate([[0], sample]))"""


training_samples = np.array(training_samples)
transposed_samples = training_samples.T
training_labels = transposed_samples[0]
#transposed_samples = np.delete(transposed_samples, 0)
training_data = transposed_samples[1:].T
print(training_data)
print(training_labels)


    
log_reg = lm.LogisticRegression()
log_reg.fit(training_data, training_labels)
print(log_reg.predict(training_data))


#print(epoched_data)
#print(epoched_data.info)
#processor.correct_for_drift(epoched_data)

#mne.viz.plot_events(epoched_data.events)
#epoched_data.plot()
#mepoched_data[9].plot_image()
#print(epoched_data)
#print(epoched_data.info)
#mne.export.export_epochs("test_e.edf", epoched_data)
#mne.export.export_raw("test.edf", epoched_data)
