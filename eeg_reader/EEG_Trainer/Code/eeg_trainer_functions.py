import os
import re
import subprocess

from pandas_ods_reader import read_ods

from kivy.uix.widget import Widget
from kivy.graphics import Line, Color

from kivy.properties import ReferenceListProperty
from kivy.properties import NumericProperty
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.properties import BooleanProperty
from kivy.properties import DictProperty
from serial.tools import list_ports

def settings_path():
    return "/home/baserad/Documents/Schoolwork/NDL/BCI_project/eeg_reader/EEG_Trainer/default_settings.ods"

def graphics_dir():
    return "/home/baserad/Documents/Schoolwork/NDL/BCI_project/eeg_reader/EEG_Trainer/Graphics"

def save_dir():
    return "/home/baserad/Documents/Schoolwork/NDL/BCI_project/eeg_reader/EEG_Trainer/Data"
def default_path():
    return str(os.getcwd())

def read_from_ods():
    return read_ods(settings_path())

def get_moves():
    return read("movements")

def get_images():
    return [os.path.join(graphics_dir(), img) for img in read("movement_images")]

def read_attribute_from_ods(attribute):
    return str(int(read_ods(settings_path())[attribute][0]))

def read(attribute):
    return read_ods(settings_path())[attribute]

#Class for storing settings.
class SettingsInfo():
    def __init__(self, trials = read("trials")[0], iterations = read("iterations")[0], seconds = read("seconds")[0], wait_time = read("wait")[0], port = None, path = default_path()):
        self.get = {
            "trials": trials,
            "iterations": iterations,
            "seconds": seconds,
            "wait": wait_time,
            "port": port,
            "path": path
        }

        self.invalid_inputs = {
            "trials": ["", "0"],
            "iterations": ["", "0"],
            "seconds": ["", "0"],
            "wait": [""],
            "port": ["", "..."]
        }

    def set_value(self, key, value):
        self.get[key] = value

    def all_input_valid(self):
        for key in self.get.keys():
            if not self.value_is_valid(key):
                return False
        return True
        
    def value_is_valid(self, key):
        if (key == "path"):
            return os.path.exists(self.get[key])
        else:
            for inv_value in self.invalid_inputs[key]:
                if self.get[key] == inv_value:
                    return False
            return True   
    

# Returns the current USB ports.

def get_ports():
    return list(list_ports.comports(include_links=True))
"""
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
"""
#Imported from https://gist.github.com/gaurav-b98/42aa141311af5f08781522cc6eec859a
class BorderBehavior(Widget):
    borders = ObjectProperty(None)
    border_origin_x = NumericProperty(0.)
    border_origin_y = NumericProperty(0.)
    border_origin = ReferenceListProperty(border_origin_x, border_origin_y)

    left_border_points = []
    top_border_points = []
    right_border_points = []
    bottom_border_points = []

    CAP = 'square'
    JOINT = 'none'

    dash_styles = {
        'dashed':
        {
            'dash_length': 10,
            'dash_offset': 5
        },
        'dotted':
        {
            'dash_length': 1,
            'dash_offset': 1
        },
        'solid':
        {
            'dash_length': 1,
            'dash_offset': 0
        }
    }

    def draw_border(self):
        line_kwargs = {
            'points': [],
            'width': self.line_width,
            'cap': self.CAP,
            'joint': self.JOINT,
            'dash_length': self.cur_dash_style['dash_length'],
            'dash_offset': self.cur_dash_style['dash_offset']
        }

        with self.canvas.after:
            self.border_color = Color(*self.line_color)
            # left border
            self.border_left = Line(**line_kwargs)

            # top border
            self.border_top = Line(**line_kwargs)

            # right border
            self.border_right = Line(**line_kwargs)

            # bottom border
            self.border_bottom = Line(**line_kwargs)

    def update_borders(self):
        if hasattr(self, 'border_left'):
            # test for one border is enough so we know that the borders are
            # already drawn
            width = self.line_width
            dbl_width = 2 * width

            self.border_left.points = [
                self.border_origin_x,
                self.border_origin_y,
                self.border_origin_x,
                self.border_origin_y +
                self.size[1] - dbl_width
            ]

            self.border_top.points = [
                self.border_origin_x,
                self.border_origin_y + self.size[1] - dbl_width,
                self.border_origin_x + self.size[0] - dbl_width,
                self.border_origin_y + self.size[1] - dbl_width
            ]

            self.border_right.points = [
                self.border_origin_x + self.size[0] - dbl_width,
                self.border_origin_y + self.size[1] - dbl_width,
                self.border_origin_x + self.size[0] - dbl_width,
                self.border_origin_y
            ]

            self.border_bottom.points = [
                self.border_origin_x + self.size[0] - dbl_width,
                self.border_origin_y,
                self.border_origin_x,
                self.border_origin_y
            ]

    def set_border_origin(self):
        self.border_origin_x = self.pos[0] + self.line_width
        self.border_origin_y = self.pos[1] + self.line_width

    def on_border_origin(self, instance, value):
        print(self.border_origin, "border origin")
        self.update_borders()
    
    def on_size(self, instance, value):
        # not sure if it's really needed, but if size is changed
        # programatically the border have to be updated
        # --> needs further testing
        if hasattr(self, 'line_width'):
            self.set_border_origin()
            self.pos = self.border_origin

    def on_pos(self, instance, value):
        # print instance, value, "pos changed"
        if hasattr(self, 'line_width'):
            self.set_border_origin()

    def on_borders(self, instance, value):
        self.line_width, self.line_style, self.line_color = value
        self.cur_dash_style = self.dash_styles[self.line_style]
        # print self.cur_dash_style, "dash_style selected"
        self.set_border_origin()
        self.draw_border()