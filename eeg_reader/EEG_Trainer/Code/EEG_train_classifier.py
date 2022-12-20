from sklearn.ensemble import RandomForestClassifier
import mne
import numpy as np

class EEGTrainClassifier():

    #Random Forest Classifier trainer
    def randomForestTrainer(self, edfFile, epoch_length):
        data = mne.io.read_raw_edf(edfFile, preload=True)

        #creating epochs
        epochs = []
        for i in range(epoch_length, len(data), epoch_length):
            epochs.append(data[i-epoch_length : i])

        #preprocess data
        low, hi = 0.2, 64
        ideal_sample_rate = 128
        for signal in epochs:
            signal.filter(low, hi)
            signal.notch_filter(np.arange(60, 240, 60))
            signal.resample(ideal_sample_rate)

        X = epochs
        y = []
        move = 0
        for epoch in epochs:
            y.append(move)
            if move == 2:
                move = 0
            else:
                move += 1

        clf = RandomForestClassifier()
        clf.fit(X,y)