import numpy as np
import serial
import time


# we might want to analyze all data at the same time -- or only pieces of the data following
# a stimuli / trigger event
# FOR a real BCI application you probably want to do feature extraction and link
# that to a classifier which makes the decision to control an external device.
class EEG_Reader:
    def __init__(self, a_port, some_baudrate = 230400, a_timeout = 0):
        self.port = a_port
        self.frequency = 10000
        self.baudrate = some_baudrate
        self.timeout = a_timeout
        self.channels = 1


        #imported variables
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


    def get_channels_amount(self):
        return self.channels

    # method comment
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
        # comment what this does
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
            
                #MSB = will be bigger than 127 when ___ and smaller when ____
                MSB  = self.input_buffer[self.cBufTail] & 0xFF
                # comment here
                if MSB > 127:
                    processed_beginning_of_frame = False
                    number_of_parsed_channels = 0

                    # comment here
                    if self.have_whole_frame():
                        while True:

                            # comment here
                            if(processed_beginning_of_frame and (MSB>127)):
                                #we have begining of the frame inside frame
                                #something is wrong
                                break #continue as if we have new frame

                            
                
                            # comment here
                            MSB  = self.input_buffer[self.cBufTail] & 0x7F
                            processed_beginning_of_frame = True
                            self.cBufTail = self.cBufTail +1

                            # comment here
                            LSB  = self.input_buffer[self.cBufTail] & 0xFF

                            if LSB>127:
                                break #continue as if we have new frame

                            LSB  = self.input_buffer[self.cBufTail] & 0x7F
                            MSB = MSB<<7
                            writeInteger = LSB | MSB
                            number_of_parsed_channels = number_of_parsed_channels+1

                            if number_of_parsed_channels > self.get_channels_amount():
                                #we have more data in frame than we need
                                #something is wrong with this frame
                                break #continue as if we have new frame
    
                            # comment here -- what does this mean
                            self.sample_buffer = np.append(self.sample_buffer,writeInteger-512)
                        

                            if self.at_end_of_frame():
                                break
                            else:
                                self.cBufTail = self.cBufTail + 1

                        # comment everything below
                    else:
                        have_data = False
                        break
                if(not have_data):
                    break

                self.cBufTail = self.cBufTail +1

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

    def test(self):
        while True:
            time.sleep(0.001)
            self.read_from_port()
            print(self.sample_buffer)


        



        
