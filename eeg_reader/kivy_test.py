import time
import EEG
import threading
import multiprocessing
import os
import re
import subprocess

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.lang.builder import Builder
from kivy.core.window import Window
from kivy.uix.stacklayout import StackLayout
from kivy.uix.anchorlayout import AnchorLayout

from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.properties import BooleanProperty


def get_ports():
    device_re = re.compile(b"Bus\s+(?P<bus>\d+)\s+Device\s+(?P<device>\d+).+ID\s(?P<id>\w+:\w+)\s(?P<tag>.+)$", re.I)
    df = subprocess.check_output("lsusb")
    devices = []
    texts = []
    for i in df.split(b'\n'):
        if i:
            info = device_re.match(i)
            if info:
                dinfo = info.groupdict()
                dinfo['device'] = '/dev/bus/usb/%s/%s' % (dinfo.pop('bus'), dinfo.pop('device'))
                devices.append(dinfo)
    return [device["device"] for device in devices]

class PortDrop(AnchorLayout):

    def __init__(self, **kwargs):
        super(PortDrop, self).__init__(**kwargs)
        self.anchor_x = 'center'
        self.anchor_y = 'top'
        self.port_is_selected = False
        self.update()
    
    def update(self):
        self.dd = DropDown()
        self.main_button = Button(text = " ... ")
        self.main_button.bind(on_release=self.dd.open)
        self.main_button.set_disabled(False)
        self.update_menu()
        self.dd.bind(on_select=lambda instance, x: setattr(self.main_button, 'text', x))
        self.main_button.size_hint_y = None
        self.main_button.height = 30
        self.add_widget(self.main_button)
    
    def select_port(self, port_text):
        self.port_is_selected = True
        self.dd.select(port_text)

    def update_menu(self):
        ports = get_ports()
        for port in ports:
            btn = Button(text = port)
            btn = Button(text=port, size_hint_y = None, height = 20)
            btn.bind(on_release=lambda btn: self.select_port(btn.text))
            self.dd.add_widget(btn)

class PortSelectorWindow(GridLayout):  

    trials_corr = BooleanProperty(True)
    iterations_corr = BooleanProperty(True)
    seconds_corr = BooleanProperty(True)
    portdropper = ObjectProperty(None, allownone= True)
    can_start = BooleanProperty(False)

    trials = 10
    seconds = 30
    iterations = 1
    no_port = " ... "

    def process_input(self, an_object):
        name = an_object.name

        if name == "trials":
            try:
                val = int(an_object.text)
                if(val == 0):
                    self.trials_corr = False
                else:
                    self.trials_corr = True
            except:
                self.trials_corr = False

        
        if name == "iterations":
            try:
                val = int(an_object.text)
                if(val == 0):
                    self.iterations_corr = False
                else:
                    self.iterations_corr = True
            except:
                self.iterations_corr = False

        
        if name == "seconds":
            try:
                val = int(an_object.text)
                if(val == 0):
                    self.seconds_corr = False
                else:
                    self.seconds_corr = True
            except:
                self.seconds_corr = False

        self.check_start()

    def check_start(self):
        if self.portdropper.port_is_selected:
            if self.seconds_corr and self.trials_corr and self.iterations_corr:
                self.can_start = True
                return

        self.can_start = False

    
    #def display_ports():
   # ports = get_ports()
    

    #for port in ports:     
   # pass




class PortSelectionApp(App):
    def build(self):
        return PortSelectorWindow()
    
    




if __name__ == '__main__':
    PortSelectionApp().run()