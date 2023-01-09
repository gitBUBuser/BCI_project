import numpy as np
import serial
import time
import os
import scipy.io.wavfile as wav
import multiprocessing
import threading
import mne 
import Code.eeg_trainer_functions as funcs
# we might want to analyze all data at the same time -- or only pieces of the data following
# a stimuli / trigger event
# FOR a real BCI application you probably want to do feature extraction and link
# that to a classifier which makes the decision to control an external device.



class EEG_Reader:
    def __init__(self, port, 
                baud_rate = float(funcs.read_ods_attr("baud_rate")),
                frequency = float(funcs.read_ods_attr("frequency")), 
                timeout = float(funcs.read_ods_attr("time_out")),
                channels = funcs.read("channels")):

        self.channels = channels
        print(channels)
        self.port = port
        self.frequency = frequency
        self.baud_rate = baud_rate
        self.timeout = timeout

        self.n_channels = len(self.channels)
        self.input_buffer = []
        self.sample_buffer = []

        self.cBufTail = 0

        self.serial_port = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)

        if self.serial_port.is_open == False:
            self.serial_port.open()

        if self.serial_port.is_open:
            print("\n Port is open. Config below:")
            print(self.serial_port, "\n")
        else:
            print("Error opening serial port. Please enter a proper port")
    
    def get_serial(self):
        return self.serial_port

    def get_data(self):
        return self.sample_buffer

    def reset_buffer(self):
        self.sample_buffer = []

    def get_channels_amount(self):
        return self.n_channels

    # checks if there is another byte in the input_buffer that is >127, so that the whole frame is in the input_buffer
    def have_whole_frame(self):
        temp_tail = self.cBufTail + 1
        # comment what this does
        while temp_tail != len(self.input_buffer):
            next_byte = self.input_buffer[temp_tail] & 0xFF
            if next_byte > 127:
                return True
            temp_tail += 1
        return False

    # checks if we are at the end of the current frame
    def at_end_of_frame(self):
        temp_tail = self.cBufTail + 1
        next_byte  = self.input_buffer[temp_tail] & 0xFF
        # if the most significant bit of the next byte is 1, return true because this implies this is a new frame
        if next_byte > 127:
            return True
        return False


    def handle_data(self, some_data):
        if len(some_data)>0:

            self.cBufTail = 0

            #if the length of some_data is larger than zero - we have data
            have_data = True
            #initiate temporary variables used in next loop
            processed_beginning_of_frame = False
            number_of_parsed_channels = 0

      
            # while we have unparsed data.
            while have_data:
            
                #MSB = takes the last 8 bits of self.input_buffer[self.cBufTail]
                MSB  = self.input_buffer[self.cBufTail] & 0xFF
                # if the most significant bit of MSB is 1
                if MSB > 127:
                    processed_beginning_of_frame = False
                    number_of_parsed_channels = 0

                    # check if the whole frame is already in the input_buffer
                    if self.have_whole_frame():
                        while True:
                            MSB  = self.input_buffer[self.cBufTail] & 0xFF

                            if(processed_beginning_of_frame and (MSB>127)):
                                #we have begining of the frame inside frame
                                #something is wrong
                                break #continue as if we have new frame

                            
                            # MSB without the most significant bit
                            MSB  = self.input_buffer[self.cBufTail] & 0x7F
                            processed_beginning_of_frame = True
                            self.cBufTail += 1

                            # next integer in the input_buffer
                            LSB  = self.input_buffer[self.cBufTail] & 0xFF

                            if LSB>127:
                                break #continue as if we have new frame

                            # LSB without the most significant bit
                            LSB  = self.input_buffer[self.cBufTail] & 0x7F
                            # bitshift MSB 7 places to the left. For example: 1111 becomes 11110000000
                            MSB = MSB<<7
                            writeInteger = LSB | MSB # bitwise or operations, simple appends the LSB behind the MSB before bitshifting
                            number_of_parsed_channels += 1

                            if number_of_parsed_channels > self.get_channels_amount():
                                #we have more data in frame than we need
                                #something is wrong with this frame
                                break #continue as if we have new frame
    
                            # appends the new frame to the sample buffer. Don't know why it does -512
                            self.sample_buffer = np.append(self.sample_buffer, writeInteger-512)

                            if self.at_end_of_frame():
                                # parsed the whole frame so break
                                break
                            else:
                                self.cBufTail += 1

                    else:
                        # there is no data anymore to parse
                        have_data = False
                        break
                if(not have_data):
                    break

                self.cBufTail += 1

                # check if there is more data in the buffer
                if self.cBufTail == len(self.input_buffer):
                    have_data = False
                    break
    def read_from_port(self):
        reading = self.get_serial().read(1024)
        if(len(reading)>0):
            reading = list(reading)
            #here we overwrite if we left some parts of the frame from previous processing 
            #should be changed             
            self.input_buffer = reading.copy()
            self.handle_data(reading)

class LiveEEGRecorder:
    def __init__(self, port):
        self.reader = EEG_Reader(port)
        self.q = multiprocessing.Queue()

    def update(self):
        time.sleep(self.reader.timeout)
        self.reader.reset_buffer()
        self.reader.read_from_port()
        buffer = self.reader.get_data()
        if len(buffer) > 0:
            self.q.put(buffer)
    
    def get_latest(self):
        latest_list = []
        while (self.q.qsize() > 0):
            latest_list.append(self.q.get())
        flat = [item for sub_list in latest_list for item in sub_list]
        return flat
        


class Counter(object):
    def __init__(self, initval = 0):
        self.val = multiprocessing.Value('i', initval)
        self.lock = multiprocessing.Lock()

    def increment(self, value):
        with self.lock:
            self.val.value += value
    
    def value(self):
        with self.lock:
            return self.val.value
        
class EEG_recorder:
    def __init__(self, port, path = os.getcwd(), record_from_start = False):

        self.reader = EEG_Reader(port)
        self.q = multiprocessing.Queue()
        self.epochs = multiprocessing.Queue()
        self.recording = multiprocessing.Event()
        self.latest = multiprocessing.Queue()
        self.sample_counter = Counter(0)

        self.path = path

        if record_from_start:
            self.start_recording()
        else:
            self.stop_recording()
        self.stop_recording()

    def start_recording(self):
        self.recording.set()

    def update(self):
        time.sleep(self.reader.timeout)
        self.reader.reset_buffer()
        self.reader.read_from_port()
        buffer = self.reader.get_data()

        if len(buffer) > 0:
            self.sample_counter.increment(len(buffer))
            self.latest.put(buffer)

        if self.recording.is_set():
            if len(buffer) > 0:
                self.q.put(buffer)


    def get_latest(self):
        latest_list = []
        while (self.latest.qsize() > 0):
            latest_list.append(self.latest.get())
        flat = [item for sub_list in latest_list for item in sub_list]
        return flat

    def get_recording_q(self):
        q_list = []
        while(self.q.qsize() > 0):
            q_list.append(self.q.get())
        flat = [item for sub_list in q_list for item in sub_list]
        return flat

    def get_epochs_annotations(self):
        epoch_dict = {}

        def channels_list_to_string(channels):
            return ','.join(channels)
        
        def channels_string_to_list(channels):
            return channels.split(",")

        def add_to_dict(epoch):
            epoch_tag = epoch[0]
            epoch_onset = epoch[1]
            channels = channels_list_to_string(epoch[2])

            if channels in epoch_dict:
                epoch_dict[channels].append([epoch_tag, epoch_onset])
            else:
                epoch_dict[channels] = [[epoch_tag, epoch_onset]]

    

        while(self.epochs.qsize() > 0):
            epoch = self.epochs.get()
            add_to_dict(epoch)

        epochs_w_durations = []

        for channels in epoch_dict.keys():
            for i in range(len(epoch_dict[channels])):
                epoch_tag = epoch_dict[channels][i][0]
                epoch_onset = float(epoch_dict[channels][i][1]) / self.reader.frequency

                if epoch_tag.startswith("stop"):
                    continue

                duration = 0

                for j in range(i, len(epoch_dict[channels])):
                    next_epoch_tag = epoch_dict[channels][j][0]

                    if next_epoch_tag == f"stop {epoch_tag}":
                        epoch_end = float(epoch_dict[channels][j][1]) / self.reader.frequency
                        duration = epoch_end - epoch_onset
                        break
            
                epochs_w_durations.append([epoch_tag, epoch_onset, duration, np.array(channels_string_to_list(channels))])

        print(epochs_w_durations)
        epochs_T = np.array(epochs_w_durations).T
        print(epochs_T)
        return mne.Annotations(epochs_T[1], epochs_T[2], epochs_T[0], ch_names=epochs_T[3])


    def stop_recording(self):
        self.recording.clear()

    def start_epoch(self, tag, channels = [""]):
        if channels == [""]:
            channels = self.reader.channels
        
        current_sample = self.sample_counter.value()
        self.epochs.put(([tag, current_sample, channels]))

    def end_epoch(self, tag, channels = [""]):
        if channels == [""]:
            channels = self.reader.channels
        current_sample = self.sample_counter.value()
        self.epochs.put([f"stop {tag}", current_sample, channels])

    def save_file(self, file_name):
        self.stop_recording()
        q_list = self.get_recording_q()
        annotations = self.get_epochs_annotations()
        print(annotations)

        save_info = np.reshape(q_list, (self.reader.n_channels, -1))
        info = mne.create_info(self.reader.channels, self.reader.frequency, ch_types="eeg")

        raw_data = mne.io.RawArray(save_info, info, copy="both")
        print(raw_data.get_data())
        raw_data.preload = True
        print("test")
        print(raw_data.times)
        print("test")
        raw_data.set_annotations(annotations)

        path = os.path.join(self.path, str(file_name))

        while os.path.exists(path + ".edf"):
            path += "I"

        mne.export.export_raw(path + ".edf", raw_data, fmt="edf")




        
