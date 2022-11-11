import numpy as np
import sklearn as sk
from sklearn import linear_model as lm
import mne as mne
import EEG_Preprocessing as preprocessing
import matplotlib.pyplot as plt

path = "/home/baserad/Documents/BYB_Recording_2022-10-14_10.12.46_edited_edited.edf"
eeg_training_data = mne.io.read_raw_edf(path)
eeg_training_data.load_data()
processor = preprocessing.EEG_processer()



eeg_training_data = processor.bandpass_filter_MNE(eeg_training_data)
event_ids = {"blink": 1, "non-target": 0}

events = mne.events_from_annotations(eeg_training_data, event_ids)[0]


print(events)

epoched_data = mne.Epochs(eeg_training_data, events, event_id=event_ids,decim=1,proj=False,preload=True)
epoched_data.drop(13)
epoched_data.apply_baseline()
evo_t = epoched_data["blink"]
evo_nt = epoched_data["non-target"]

blinks = []
others = []

#Transform epoch into feature vector
average_interval = 0
samples_n = 600
training_samples = []

for epoch in evo_t:
    sample = []
    epoch_size = epoch.shape[1]
    step = int(epoch_size / samples_n)

    for i in range(0,epoch_size, step):
        if len(sample) == samples_n:
            continue
        else:
            interval_behind = average_interval
            interval_above = average_interval

            if i - average_interval < 0:
                interval_behind = i

            if i + average_interval > epoch_size:
                interval_above = epoch_size - i

            avg = 0

            for j in range(i - interval_behind - 1, i + interval_above):
                avg += epoch[0][j]

            avg /= (interval_behind + interval_above + 1)

            sample.append(avg)

    if len(sample) == samples_n:
        training_samples.append(np.concatenate([[1], sample]))



for epoch in evo_nt:
    sample = []
    epoch_size = epoch.shape[1]
    step = int(epoch_size / samples_n)

    for i in range(0, epoch_size, step):
        if len(sample) == samples_n:
            continue
        else:
            interval_behind = average_interval
            interval_above = average_interval

            if i - average_interval < 0:
                interval_behind = i

            if i + average_interval > epoch_size:
                interval_above = epoch_size - i

            avg = 0

            for j in range(i - interval_behind - 1, i + interval_above):
                avg += epoch[0][j]

            avg /= (interval_behind + interval_above + 1)

            sample.append(avg)

    if len(sample) == samples_n:
        training_samples.append(np.concatenate([[0], sample]))
        

t = []
for i in range(20):
    for sample in training_samples:
        t.append(sample)



training_samples = np.array(t)
transposed_samples = training_samples.T
training_labels = transposed_samples[0]
print(training_labels)
training_data = transposed_samples[1:].T

"""fig, ax = plt.subplots()
for i in range(epoched_data.__len__()):
    if training_labels[i] == 1:
        ax.plot(training_data[i], label = training_labels[i],color = "red")
    else:
        ax.plot(training_data[i], label = training_labels[i],color = "blue")
fig.legend()
plt.show()
"""

log_reg = lm.LogisticRegression()
print(training_data.shape)
log_reg.fit(training_data, training_labels)
print(log_reg.predict(training_data))


