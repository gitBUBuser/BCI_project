import numpy as np
import serial
import time
import os
import scipy.io.wavfile as wav
import multiprocessing
import threading
import mne 
# we might want to analyze all data at the same time -- or only pieces of the data following
# a stimuli / trigger event
# FOR a real BCI application you probably want to do feature extraction and link
# that to a classifier which makes the decision to control an external device.



class EEG_Reader:
    def __init__(self, a_port, some_baudrate = 230400, a_timeout = 0.00025):
        self.port = a_port
        self.frequency = 10000
        self.baudrate = some_baudrate
        self.timeout = a_timeout
        self.channels = 1
        self.input_buffer = []
        self.sample_buffer = []
        self.cBufTail = 0

        self.serial_port = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

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
        return self.channels

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

class EEG_recorder:
    def __init__(self, a_port):
        self.port = a_port
        self.reader = EEG_Reader(a_port)
        self.sampling_interval = self.reader.timeout
        self.frequency = self.reader.frequency
        self.q = multiprocessing.Queue()
        self.recorded_data = []
        self.recording = multiprocessing.Event()

        self.stop_recording()

    def start_recording(self):
        self.recording.set()


    def update(self):
        time.sleep(self.sampling_interval)
        self.reader.reset_buffer()
        self.reader.read_from_port()

        if self.recording.is_set():
            buffer = self.reader.get_data()
            if len(buffer) > 0:
                self.q.put(buffer)
            

    def stop_recording(self):
        self.recording.clear()

    def save_file(self, file_name):
        self.stop_recording()
        q_list = []
        print(self.q.qsize())
        while(self.q.qsize() > 0):
            q_list.append(self.q.get())
        print(self.q.qsize())

        data = list(q_list)
        
        flat_data = [item for sub_list in data for item in sub_list]

        save_info = np.array(flat_data)
        print(save_info.size)
        save_info = np.reshape(save_info, (1, -1))
        print(save_info.size)
        print("printed size")
        info = mne.create_info(["Cz"], self.frequency)
        raw_data = mne.io.RawArray(save_info, info)

        path = os.getcwd()  + "/" + str(file_name)
        mne.export.export_raw(path, raw_data, fmt="edf")


# Class for preprocessing EEG signals.
class EEG_processer:
    def __init__(self):
        self.filter_band = (1.5, 16)
        self.inteval = (-0.1, 0)


    def bandpass_filter_MNE(self, some_data):
        return some_data.filter(self.filter_band[0], self.filter_band[1], method="icir")

    def preprocessed_to_epoch(self, preprocessed_data, decimate=10, baseline_ival=(-.2, 0)):
        class_ids = { "Left": 1, "Right": 2, "StartShoot": 3, "StopShoot": 4}

        events = mne.events_from_annotations(preprocessed_data, event_id=class_ids)[0]
        epo_data = mne.Epochs(preprocessed_data, events, event_id=class_ids,
                              baseline=baseline_ival, decim=decimate,
                              reject=reject, proj=False, preload=True)
        return epo_data

    def correct_for_drift(self, some_data):
        some_data.apply_baseline(self.inteval)
        



        
