from sklearn.ensemble import RandomForestClassifier
from invaders.game import *
import numpy as np

class EEGGame():
    #function for processing the data. the clf parameter is the RandomForestClassifier that is fitted with training data
    def process_data(self, plotted_seconds, downsampled_rate, clf):
        global recorder
        self.sample()

        plotted_values = plotted_seconds * recorder.frequency
        self.plot_buffer_full = self.plot_buffer_full[-int(plotted_values):]

        #preprocess data
        downsampled_plot = self.plot_buffer_full[0::int(recorder.frequency / downsampled_rate)]

        low, hi = 0.2, 64
        ideal_sample_rate = 128
        for signal in downsampled_plot:
            signal.filter(low, hi)
            signal.notch_filter(np.arange(60, 240, 60))
            signal.resample(ideal_sample_rate)

        move = clf.predict(downsampled_plot)
        if move == 1:
            goRight()
        elif move == 2:
            goLeft()
        else:
            doNothing()