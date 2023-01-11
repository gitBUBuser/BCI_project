from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from game import *
import numpy as np
from EEG.Preprocesser import DataHandler
from EEG.EEG import LiveEEGRecorder
from statistics import mean
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import time
#from eeg_reader.EEG_Trainer.Code.EEG import LiveEEGRecorder
from statistics import mean, stdev
from game import Game
import multiprocessing
from EEG.Preprocesser import BlankObj
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import VALID_METRICS 
from sklearn.metrics import confusion_matrix,plot_confusion_matrix

recorder = None
stop_update = multiprocessing.Event()


def update_recorder():
    global recorder
    while True:
        recorder.update()
        if stop_update.is_set():
            break

    
recorder_loop_T = multiprocessing.Process(target=update_recorder)


class EEGGame():
    def __init__(self):
        self.time_saved = 5
        self.extra_time_saved = 10


        clear = lambda: os.system('clear')
        clear()

        self.handler = DataHandler([1, 40], 500)
        self.user_data = []

        start = True
        while start:
            print(self.handler)
            print()
            print("Please provide the path to your training data: ")
            path = str(input())
            try: 
                self.handler.load_file(path)
                self.handler.detrend()
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
            #try:
            global recorder
            recorder = LiveEEGRecorder(self.port)
            self.raw_data = np.zeros(int(recorder.reader.frequency * (self.time_saved + self.extra_time_saved))).tolist()
            self.sampling_rate = recorder.reader.frequency
            start = False
#            except:
 #               print(" >> There was an error! please confirm that the port is valid")
        """
        
        neighs = np.arange(2,15)
        trees = 100
        a = np.arange(0.1, 10, 0.5)
        n_topics = np.arange(1,150,1)
        param_grid=dict(n_estimators = [100])
        best_params = {}

        self.handler.segment(2)
        print(self.handler)
        print()
        possible_band_amounts = np.arange(3, 20)

        best_band_amount = None
        best_score = 0
        best_overall_score = None
        best_params = None

        for i in possible_band_amounts:
            bands = self.handler.get_bands(2, 33, i)
            X, y = self.handler.bandpowers(bands)
   

            clf_0 =  RandomForestClassifier()
            clf = GridSearchCV(clf_0, param_grid, cv=5, scoring = "accuracy", verbose = 1)

        
           
            scores = []
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=32)
            X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
            grid_search = clf.fit(X_train, y_train)
            print(grid_search.best_params_)
            print(grid_search.best_score_)
                

            if grid_search.best_score_ > best_score:
                clf_real = RandomForestClassifier().set_params(**grid_search.best_params_)
                best_params = grid_search.best_params_
                clf_real.fit(X_train, y_train)
                best_overall_score = clf_real.score(X_test, y_test)
                best_score = grid_search.best_score_
                best_band_amount = i
        
        print("BEST band amount: ",   best_band_amount)
        print("REAL accuracy of model: ", best_overall_score * 100, "%")
        print()
        print("-------------------------------------------------------")
        print()
        print('\nOverall Accuracy:',
            best_score*100, '%')
        print()
        print()
        print("press 'enter' to continue...")
        input()
        self.bands = self.handler.get_bands(2, 33, best_band_amount)

        print(self.bands)
        X, y = self.handler.bandpowers(self.bands)
        input()
        self.clf = RandomForestClassifier()
        self.clf.set_params(**best_params)
        self.clf.fit(X, y)

        recorder_loop_T.start()
        self.game = Game()


    def update(self):
        global recorder
        X = self.handle_data(recorder.get_latest())
        log_probas = self.clf.predict_proba(X)
        print(log_probas)
        print("score: ")
        prediction = self.clf.predict(X)

        if prediction == 0:
            self.game.goLeft()
        if prediction == 1:
            self.game.goRight()
        self.game.update()
    
    def handle_data(self, data):
        self.raw_data.extend(data)
        saved_seconds = self.time_saved + self.extra_time_saved
        saved_values = (saved_seconds * self.sampling_rate)
        self.raw_data = self.raw_data[-int(saved_values):]
        filtered_raw = self.handler.filter_and_resample(self.raw_data, saved_seconds)
        voi = self.time_saved * self.handler.sampling_rate
        self.handler.set_signal(filtered_raw[-int(voi):])
        self.handler.segment(2)
        X, y = self.handler.bandpowers(self.bands)
        X,y = self.handler.average_series(X, interval = 2)
        return X


if __name__ == "__main__":
    game = EEGGame()
    while True:
        game.update()
    stop_update.set()