from Preprocesser import DataHandler
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = "/home/baserad/Documents/Schoolwork/NDL/BCI_project/EEG/EEG_Trainer/sven_training.edf"
    handler = DataHandler([1, 40], 500)
    handler.load_file(path)

    handler.detrend()
    bands = handler.get_bands(2, 32, 10)
    X, y = handler.bandpowers(bands)

    fig, ax = plt.subplots()

    signals_by_labels = {}

    for i in range(len(list(dict.fromkeys(y)))):
        signals_by_labels[handler.label_names[i]] = [X[j] for j in range(len(X)) if y[j] == i]


    for class_label, signals in signals_by_labels.items():
        ax.plot(bands, signals, label = class_label)

    plt.legend()
    plt.show()
