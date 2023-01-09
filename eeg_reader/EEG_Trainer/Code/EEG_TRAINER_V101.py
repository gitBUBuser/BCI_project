from kivy.uix.pagelayout import PageLayout
from kivy.uix.screenmanager import Screen, ScreenManager, FadeTransition


import time
import threading
import multiprocessing
import os
import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.lang.builder import Builder
from kivy.core.window import Window
from kivy.uix.stacklayout import StackLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout

from kivy.properties import ObjectProperty
from kivy.properties import StringProperty

from kivy.properties import BooleanProperty
from kivy.properties import ReferenceListProperty
from kivy.properties import NumericProperty
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.graphics import Line, Color
from kivy.uix.image import Image

import Code.EEG as EEG
import Code.eeg_trainer_functions as funcs

recorder = None
stop_update = multiprocessing.Event()


def update_recorder():
    global recorder
    while True:
        recorder.update()
        if stop_update.is_set():
            break

    
recorder_loop_T = multiprocessing.Process(target=update_recorder)


#Movements we wish to train on.
movements = funcs.get_moves()
#Images corresponding to training moves.
movement_images = funcs.get_images()
wait_sign_path = os.path.join(funcs.graphics_dir(), "WaitSign.png")

#Class to allow FigureCanvasKivyAgg init in kv language.
class BetterPlot(FigureCanvasKivyAgg):
    def __init__(self, plot = plt.gcf(), **kwargs):
        super().__init__(plot, **kwargs)

# TrainerWindow
class EEGTrainerWindow(BoxLayout):
    eeg_plot = ObjectProperty()

    time_display_str = StringProperty()
    trial_display_str = StringProperty()
    iteration_display_str = StringProperty()

    instructions_image_source = StringProperty()
    move_index_text = StringProperty()
    instruction_text = StringProperty()

class EEGTrainer(Screen):
    user_interface = ObjectProperty()

    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)

        self.movements = movements
        self.movement_images = movement_images

        #Timer variables.
        self.iteration_timer = 0
        self.wait_timer = 0
        self.original_wait_timer = 0
        self.run_time = 0
         #Variables corresponding to stage of training
        self.current_trial = 1
        self.current_move = 0
        self.current_iteration = 1

        #Buffer used for plotting online EEG data
        
        self.plot_buffer_full = list(np.zeros(50000))
        self.target_condition = False
        self.movement_just_started = False
        self.movement_just_ended = True
   
        #True first main loop, otherwise False.
        self.just_started = True

    def stop(self):
        global recorder
        stop_update.set()

        recorder.save_file(self.subject)
        print("saved_file")
        print("stopped rec")
        recorder.get_recording_q()
        print("got Q")
        recorder.get_latest()
        print("got latest")

        recorder_loop_T.join()
        print("stopped thread")
  

    def initialize(self, settings):
        global recorder

        self.subject = settings.get["subject"]
        self.port = settings.get["port"]
        self.path = settings.get["path"]

        self.port = self.port.split(" ")[0]

        self.trials = int(settings.get["trials"])
        self.iterations = int(settings.get["iterations"])
        self.iteration_time = float(settings.get["seconds"])
        self.wait_time = float(settings.get["wait"])
        self.original_wait_time = self.wait_time
        self.start_wait_time = self.wait_time + 10
        
        recorder = EEG.EEG_recorder(self.port, self.path, record_from_start=True)
        recorder.start_recording()
        recorder_loop_T.start()
        Clock.schedule_interval(self.main_loop, .1)

    #Provides user with instructions while waiting.
    def instructions_on_wait(self):
        self.user_interface.instruction_text = ".. Await instructions .."
        self.user_interface.instructions_image_source = wait_sign_path

    #Provides instuction for user movement
    def instructions_on_move(self):
        self.user_interface.instruction_text = self.movements[self.current_move]
        self.user_interface.instructions_image_source = self.movement_images[self.current_move]
    
    #Updates instruction labels in application.
    def update_labels(self):
        self.user_interface.trial_display_str = self.get_trial_text()
        self.user_interface.iteration_display_str = self.get_iterations_text()
        self.user_interface.move_index_text = self.get_move_index_text()
    
    ### HELPER FUNCTIONS FOR TESTING ####
    def sample(self):
        global recorder
        for ele in recorder.get_latest():
            self.plot_buffer_full.append(ele)

    #### END ###

    # Graphs the EEG data from the user.
    def graph_plot(self):
        global recorder
        self.sample()
        freq = recorder.reader.frequency


        plotted_seconds = 5
        plotted_values = plotted_seconds * freq
        self.plot_buffer_full = self.plot_buffer_full[-int(plotted_values):]
        downsampled_rate = 250
        downsampled_plot = self.plot_buffer_full[0::int(freq / downsampled_rate)]
        xi = np.arange(-plotted_seconds, 0, 1 / downsampled_rate)
        plt.clf()
        plt.ylim(-500, 500)
        plt.plot(xi, downsampled_plot, linewidth=1, color='royalblue')
        self.user_interface.eeg_plot.draw()
        
    
    #Updates the current training instructions / training phase
    def update_instructions(self):
        if not self.just_started:
            if self.current_iteration > self.iterations - 1:
                self.current_iteration = 1
                self.current_move += 1
            else: 
                self.current_iteration += 1
                return
        
            if self.current_move > len(self.movements) - 1:
                self.current_move = 0
                self.current_trial += 1

            if self.current_trial > self.trials - 1:
                #end training
                pass
    
    ##### METHODS FOR RETURNING DISPLAY VALUES ####
    def set_display_time_text(self, time, max_time):
        self.user_interface.time_display_str = f'{np.round(time, 1)}s / {max_time}'

    def get_trial_text(self):
        return f'{self.current_trial} / {self.trials}'
    
    def get_iterations_text(self):
        return f'{self.current_iteration} / {self.iterations}'

    def get_move_index_text(self):
        return f'{self.current_move + 1} / {len(self.movements)}'
    
    #### END OF SECTION ####

      # Main loop of the trainer application. nand is the time since last update.
    def main_loop(self, nand):
        global recorder
        self.run_time += nand
        self.graph_plot()
        
        if self.just_started:
            self.wait_time = self.start_wait_time
        else:
            self.wait_time = self.original_wait_time

        if self.wait_timer > self.wait_time:
            self.target_condition = True
            self.movement_just_started = True
            self.wait_timer = 0
            
        if self.movement_just_started:
            self.movement_just_started = False
            self.update_instructions()
            self.update_labels()
            self.instructions_on_move()
            self.just_started = False
            recorder.start_epoch(self.movements[self.current_move])

        if self.target_condition:
            self.iteration_timer += nand
            self.set_display_time_text(self.iteration_timer, self.iteration_time)

            if self.iteration_timer > self.iteration_time:
                self.iteration_timer = 0
                self.target_condition = False
                self.movement_just_ended = True
        else:
            self.wait_timer += nand
            self.set_display_time_text(self.wait_timer, self.wait_time)

        if self.movement_just_ended:
            if not self.just_started:
                recorder.end_epoch(self.movements[self.current_move])

            self.movement_just_ended = False
            self.update_labels()
            self.instructions_on_wait()
           
