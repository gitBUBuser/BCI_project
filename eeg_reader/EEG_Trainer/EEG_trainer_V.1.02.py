import Code.EEG_settings_editor as settings_editor
import Code.EEG_TRAINER_V101 as trainer
import Code.eeg_trainer_functions as funcs

import os

from kivy.uix.pagelayout import PageLayout
from kivy.uix.screenmanager import Screen, ScreenManager, FadeTransition
from kivy.app import App
from pandas_ods_reader import read_ods
from kivy.lang import Builder

class EEGTrainerApp(App):
    def build(self):
        self.sm = ScreenManager()
        
        self.selection_screen = settings_editor.SelectionScreen(name = "menu")
        self.trainer_screen = trainer.EEGTrainer(name = "trainer")
        
        self.selection_screen.user_interface.file_path = os.getcwd()

        self.sm.add_widget(self.trainer_screen)
        self.sm.add_widget(self.selection_screen)
        self.sm.current = "menu"
        return self.sm


    def on_start(self):
        self.selection_screen.user_interface.portdropper.init_dropdown(funcs.get_ports())

    def start_trainer(self):
        self.sm.switch_to(self.trainer_screen)
        self.trainer_screen.initialize(self.selection_screen.user_interface.settings)
        print("started_trainer")

    def on_request_close(self, *args):
        print("tried to stop")
        self.selection_screen.stop()
        self.trainer_screen.stop()
    
    def on_stop(self):
        print("tried to stop throught stop")
        try:
            self.selection_screen.stop()
            print("stopped selection")
        except:
            pass
        """
        try: 
            self.trainer_screen.stop()
            print("stopped trainer")
        except:
            pass
        """
        self.trainer_screen.stop()
        print("stopped trainer")
    
if __name__ == "__main__":
    EEGTrainerApp().run()
