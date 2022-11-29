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



movements = ["Default", "Right hand right", "Left hand left"]
#recorder = EEG.EEG_recorder("/dev/ttyACM0")

"""def update_recorder():
    while(True):
        recorder.update()

update = multiprocessing.Process(target=update_recorder)
update.start()"""

default_trials = 10
default_iterations = 3

n_trials = 0
n_iterations = 0


def get_ports():
    device_re = re.compile(b"Bus\s+(?P<bus>\d+)\s+Device\s+(?P<device>\d+).+ID\s(?P<id>\w+:\w+)\s(?P<tag>.+)$", re.I)
    df = subprocess.check_output("lsusb")
    devices = []
    for i in df.split(b'\n'):
        if i:
            info = device_re.match(i)
            if info:
                dinfo = info.groupdict()
                dinfo['device'] = '/dev/bus/usb/%s/%s' % (dinfo.pop('bus'), dinfo.pop('device'))
                devices.append(dinfo)
    return devices

    

class EEG_trainer():
    pass

class SelectionLayout(GridLayout):
    def __init__(self):
        super().__init__()
        self.rows = 2
        self.add_widget(Label(text = "Welcome to EEG trainer for NDL : V.1.0"))

        dd_frame = GridLayout()
        dd_frame.rows = 2
        dd_frame.padding=(50,50)
        dd = DropDown()

        ports = get_ports()
        port_texts = []

        for i in range(len(ports)):
            port_texts.append(f"{ports[i]['tag']} - {ports[i]['device']}")

        for port in port_texts:
            btn = Button(text=port, size_hint_y = None, height=20)
            btn.bind(on_release=lambda btn: dd.select(btn.text))
            dd.add_widget(btn)
        
        main_button = Button(text=" ... ", size_hint=(None, None), height = 20)

        main_button.bind(on_release=dd.open)
        dd.bind(on_select=lambda instance, x: setattr(main_button, 'text', x))

        dd_frame.add_widget(Label(text="Select port: ", size_hint = (0,0)))
        dd_frame.add_widget(main_button)
        dd_frame.add_widget(dd)

        self.add_widget(dd_frame)


class PortSelectionApp(App):
    def build(self):
        return SelectionLayout()

        

def init_trainer():
    PortSelectionApp().run()
    
    

def constuct_trainer():
    def yes_no_viable(value):
        if value == "y" or value == "Y":
            return True
        if value == "n" or value == "N":
            return True
        return False
    
    def is_viable_input(value):
        if value == "-h":
            print_help_screen()
            print_start()
            return False
        if value == "d":
            return True
        try:
            int(value)
            return True
        except:
            return False

    def clear_terminal():
        os.system("cls" if os.name == "nt" else "clear")

    def print_start():
        clear_terminal()
        print()
        print("--- welcome to EEG_trainer for NDL : V.1.0 ---")
        print()
        print("type -h for instructions!")

        print()
        print()
        if trials != None:
            print(f"Trials: {trials}" )
        if iterations != None:
            print(f"Iterations: {iterations}")
        print("______________________________________________________________________________")
        print("______________________________________________________________________________")

        print()   

    def print_help_screen():
        clear_terminal()
        print(" ---- HELP ----")
        print()
        print()
        print(f" - trials: -- Denotes the number of trials. Basically, how many times you will have to think!")
        print(f" - iterations -- Denotes how many times the trainer will repeat an instruction/task per trial")
        print()
        print(" --- ")
        print()
        input("Press any key to continue... ")
    

    trials = None
    iterations = None
    value = None

    print_start()

    while not is_viable_input(value):
        value = input("Enter your wanted number of trials (d=default): ")
        print("invalid input..")
        print()

    if(value == "d"):
        trials = default_trials
    else:
        trials = value
    print_start()
    value = None

    while not is_viable_input(value):
        value= input("Enter your wanted number of iterations (d=default): ")

    if(value == "-h"):
        print_help_screen()
    if(value == "d"):
        iterations = default_iterations
    else:
        iterations = value
    
    n_trials = trials
    n_iterations = iterations

    print_start()

    val = "x"
    while(not yes_no_viable(val)):
        val = input("Would you like to change your values? (y/n): ")

    if(val == "n"):
        init_trainer()
    if(val == "y"):
        constuct_trainer()



def main():
    constuct_trainer()




   # update.join()
    

if __name__ == "__main__":
    main()

