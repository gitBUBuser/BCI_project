import os
from directkeys import PressKey, ReleaseKey, LEFT, RIGHT, ENTER, SPACE
import time

def main():
    os.startfile("invaders.exe")
    time.sleep(1)
    PressKey(ENTER)
    ReleaseKey(ENTER)
    time.sleep(1)

    #read EEG, run algorithm and call move functions

def goLeft():
    PressKey(LEFT)
    time.sleep(0.4)
    ReleaseKey(LEFT)

def goRight():
    PressKey(RIGHT)
    time.sleep(0.4)
    ReleaseKey(RIGHT)

def startShooting():
    PressKey(SPACE)

def stopShooting():
    ReleaseKey(SPACE)

if __name__ == "__main__":
    main()