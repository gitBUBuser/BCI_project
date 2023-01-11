import os
from pynput.keyboard import Key, Controller
import time
import subprocess, sys
from pynput.keyboard import Key
from sys import platform
from SpaceInvadersMasters.spaceinvaders import SpaceInvaders

class Game():
    def __init__(self):
        self.keyboard = Controller()
        self.game = SpaceInvaders()
        time.sleep(0.3)
        self.keyboard.press(Key.enter)
        self.keyboard.release(Key.enter)

    def doNothing(self):
        self.keyboard.release(Key.left)
        self.keyboard.release(Key.right)
    
    def goLeft(self):
        self.keyboard.release(Key.right)
        self.keyboard.press(Key.left)
    
    def goRight(self):
        self.keyboard.release(Key.left)
        self.keyboard.press(Key.right)
    
    def update(self):
        self.game.main()