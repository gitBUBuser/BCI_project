import EEG
import numpy as np
import scipy.io

port = ""
reader = EEG.EEG_Reader(a_port)
sampling_interval = 0.001

recorded_data = []
recording = False

def start_recording():
    recording = True
    while(recording):
        record()

def stop_recording():
    recording = False

def record():
    time.sleep(sampling_interval)
    reader.read_from_port()
    recorded_data.append(reader.get_data())

def save_file(file_name):
    stop_recording()
    
    frequency = 10000
    save_info = np.array(recorded_data)
    wavfile.write("file_name", frequency, save_info)
    recorded_data.clear()