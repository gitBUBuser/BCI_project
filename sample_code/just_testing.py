import mne
from EEG_preprocessing import DataHandler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import numpy as np
from statistics import mean, stdev
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    path = "/home/baserad/Documents/Schoolwork/NDL/BCI_project/eeg_reader/EEG_Trainer/Min_lilla_hjärna.edf"
    handler = DataHandler([1, 40], 240)
    handler.load_file(path)

    handler.segment(1)

    print(handler)
    bands = handler.get_bands(2, 35, 3)
    X, y = handler.bandpowers(bands)
    X, y = handler.average_series(X, y, 3)

    X, y = np.array(X), np.array(y)
    print(X)
    print(y)

    
    log_reg = RandomForestClassifier()
    lda = LinearDiscriminantAnalysis()

   
    lst_accu_stratified = []
    lst_accu_log_reg = []

    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        log_reg.fit(X_train, y_train)
        lda.fit(X_train, y_train)
        lst_accu_stratified.append(lda.score(X_test, y_test))
        lst_accu_log_reg.append(log_reg.score(X_test, y_test))


    
    print('List of possible accuracy:', lst_accu_stratified)
    print('\nMaximum Accuracy That can be obtained from this model is:',
	    max(lst_accu_stratified)*100, '%')
    print('\nMinimum Accuracy:',
	    min(lst_accu_stratified)*100, '%')
    print('\nOverall Accuracy:',
	    mean(lst_accu_stratified)*100, '%')
    print('\nStandard Deviation is:', stdev(lst_accu_stratified))

    print('List of possible accuracy:', lst_accu_log_reg)
    print('\nMaximum Accuracy That can be obtained from this model is:',
	    max(lst_accu_log_reg)*100, '%')
    print('\nMinimum Accuracy:',
	    min(lst_accu_log_reg)*100, '%')
    print('\nOverall Accuracy:',
	    mean(lst_accu_log_reg)*100, '%')
    print('\nStandard Deviation is:', stdev(lst_accu_log_reg))


    


    

    
    
    

