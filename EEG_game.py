from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from invaders.game import *
import numpy as np
from sample_code.EEG_preprocessing import DataHandler
from statistics import mean
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
#from eeg_reader.EEG_Trainer.Code.EEG import LiveEEGRecorder
from statistics import mean, stdev
"""
recorder = None
stop_update = multiprocessing.Event()


def update_recorder():
    global recorder
    while True:
        recorder.update()
        if stop_update.is_set():
            break

    
recorder_loop_T = multiprocessing.Process(target=update_recorder)

"""
class EEGGame():
    def __init__(self):
        self.time_saved = 4
        self.extra_time_saved = 4


        clear = lambda: os.system('clear')
        clear()

        self.handler = DataHandler([1, 40], 240)
        self.user_data = []

        start = True
        while start:
            print(self.handler)
            print()
            print("Please provide the path to your training data: ")
            path = str(input())
            try: 
                self.handler.load_file(path)
                print(self.handler)
                start = False
                clear()
            except:
                print(" >> There was an error! Ensure that the path is valid")
                input()
                clear()
        """
        start = True
        while start:
            print(self.handler)
            print()
            print("Please provide the port of your EEG device: (find port in trainer if unsure)")
            self.port = str(input())
            try:
                global recorder
                recorder = LiveEEGRecorder(self.port)
                self.raw_data = np.zeros(recorder.reader.frequency * (self.time_saved + self.extra_time_saved)).tolist()
                self.sampling_rate = recorder.reader.frequency
                start = False
            except:
                print(" >> There was an error! please confirm that the port is valid")
        """

        skf = StratifiedKFold(n_splits=10)
        self.handler.segment(2)

        possible_band_amounts = np.arange(3, 15)

        best_band_amount = None
        best_score = np.zeros(1).tolist()
        best_overall_score = None

        for i in possible_band_amounts:
            bands = self.handler.get_bands(2, 35, i)
            X, y = self.handler.bandpowers(bands)
            X, y = self.handler.average_series(X, y, 3)

            lda = LinearDiscriminantAnalysis()
            scores = []
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
            for train_index, test_index in skf.split(X_train, y_train):
                x_train_fold, x_test_fold = X_train[train_index], X_train[test_index]
                y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
                lda.fit(x_train_fold, y_train_fold)
                scores.append(lda.score(x_test_fold, y_test_fold))

            if mean(scores) > mean(best_score):
                lda.fit(X_train, y_train)
                best_overall_score = lda.score(X_test, y_test)
                best_score = scores
                best_band_amount = i
        
        print("BEST band amount: ",   best_band_amount)
        print("REAL accuracy of model: ", best_overall_score * 100, "%")
        print()
        print("-------------------------------------------------------")
        print()
        print('List of possible accuracy:', best_score)
        print('\nMaximum Accuracy That can be obtained from this model is:',
	        max(best_score)*100, '%')
        print('\nMinimum Accuracy:',
            min(best_score)*100, '%')
        print('\nOverall Accuracy:',
            mean(best_score)*100, '%')
        print('\nStandard Deviation is:', stdev(best_score))
        print()
        print()
        print()
        print("press 'enter' to continue...")
        input()
        self.bands = self.handler.get_bands(2, 35, best_band_amount)
        X, y = self.handler.bandpowers(bands)
        self.clf = LinearDiscriminantAnalysis()
        self.clf.fit(X, y)
        breakpoint()
        recorder_loop_T.start()


    def update(self):
        time.sleep(0.1)
        global recorder
        X, y = self.handle_data(recorder.get_latest())
        prediction = self.clf.predict(X)
        print(self.handler.label_names[prediction])

    
    def handle_data(self, data):
        self.raw_data.extend(data)
        saved_seconds = self.time_saved + self.extra_time_saved
        saved_values = (saved_seconds * self.sampling_rate)
        self.raw_data = self.raw_data[-int(saved_values):]
        filtered_raw = self.handler.filter_and_resample(self.raw_data)
        voi = self.time_saved * self.handler.sampling_rate
        self.handler.set_signal(filtered_raw[-int(voi)])
        return handler.bandpowers(self.best_band_amount)


if __name__ == "__main__":
    game = EEGGame()
    while True:
        game.Update()
    stop_update.set()