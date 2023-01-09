import os
from pynput.keyboard import Key, Controller
import time

class Game():
    def __init__():
        self.keyboard = Controller()
        os.startfile(os.getcwd() + "\\invaders\\invaders.exe")
        time.sleep(1)
        self.keyboard.press('ENTER')
        time.sleep(1)
        self.keyboard.release('ENTER')

    def doNothing(self):
        self.keyboard.release('LEFT')
        self.keyboard.release('RIGHT')
    
    def goLeft(self):
        self.keyboard.release('RIGHT')
        self.keyboard.press('LEFT')
    
    def goRight(self):
        self.keyboard.release('LEFT')
        self.keyboard.press('RIGHT')