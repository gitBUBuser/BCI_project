import time
import threading
import multiprocessing
import os
import re
import subprocess

from pandas_ods_reader import read_ods
from kivy.uix.pagelayout import PageLayout
from kivy.uix.screenmanager import Screen, ScreenManager, FadeTransition
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.properties import ReferenceListProperty
from kivy.properties import NumericProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.lang.builder import Builder
from kivy.core.window import Window
from kivy.uix.stacklayout import StackLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics import Line, Color
from kivy.uix.image import Image
import Code.eeg_trainer_functions as funcs

from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.properties import BooleanProperty
from kivy.properties import DictProperty
class ErrorLabel(Label):
    visible = BooleanProperty()

class CustomDropDown(AnchorLayout):
    def __init__(self, options=[], **kwargs):
        super(CustomDropDown, self).__init__(**kwargs)
        self.dd = DropDown()
        self.main_button = Button()
        self.main_button.bind(on_release=self.dd.open)
        self.dd.bind(on_select=lambda instance, x: setattr(self.main_button, 'text', x))
        self.add_widget(self.main_button)
        self.size_hint_y = None
        self.size_hint_min_x = None

        try:
            self.init_dropdown(options)
        except:
            pass

    def select_option(self,element_text):
        self.dd.select(element_text)

        try:
            self.root.set_n_check_value("port", element_text)
            self.root.check_start()
        except:
            pass

        self.exapand_menu()

    def exapand_menu(self):
        self.dd.clear_widgets()
        options_size = len(self.options) + 1
        wanted_main_scalar = 1.2
        max_size_main = 40
        max_size_button = 30
    
        button_s = self.height / options_size


        main_button_s = button_s * wanted_main_scalar
        if main_button_s > max_size_main:
            main_button_s = max_size_main


        new_button_s = (self.height - main_button_s) / options_size

        if new_button_s > max_size_button:
            new_button_s = max_size_button

        self.main_button.size_hint_y = None
        self.main_button.height = main_button_s

        for option in self.options:
            btn = Button(text = str(option), size_hint_y = None, height = new_button_s)
            btn.bind(on_release=lambda btn: self.select_option(btn.text))
            if self.main_button.text == btn.text:
                btn.disabled = True

            self.dd.add_widget(btn)

    def init_dropdown(self, options):
        self.options = options
        try:
            self.select_option(str(options[0]))
        except:
            self.select_option("...")

class PortSelectorWindow(GridLayout):  
    value_corrs = DictProperty({
        "trials": True,
        "iterations": True,
        "seconds": True,
        "port": True,
        "wait": True,
        "path": True,
        "subject": True
    })

    portdropper = ObjectProperty(None, allownone= True)
    can_start = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = funcs.SettingsInfo()

    def process_text_input(self, an_object):
        name = an_object.name
        self.set_n_check_value(name, an_object.text)

        print(self.value_corrs)
        self.check_start()
        print(self.can_start)

    def set_n_check_value(self, key, value):
        self.settings.set_value(key, value)
        self.value_corrs[key] = self.settings.value_is_valid(key)

    def check_start(self):
        for value in self.value_corrs.values():
            if value == False:
                self.can_start = False
                return
        self.can_start = True        

class SelectionScreen(Screen):
    user_interface = ObjectProperty(None, allownone= True)

    def stop(self):
        pass