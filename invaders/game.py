import os
from directkeys import PressKey, ReleaseKey, LEFT, RIGHT, ENTER, SPACE
import time

def startGame():
    os.startfile(os.getcwd() + "\\invaders\\invaders.exe")
    time.sleep(1)
    PressKey(ENTER)
    ReleaseKey(ENTER)
    time.sleep(1)

    PressKey(SPACE)
    #read EEG, run algorithm and call move functions

def doNothing():
    ReleaseKey(LEFT)
    ReleaseKey(RIGHT)

def goLeft():
    ReleaseKey(RIGHT)
    PressKey(LEFT)

def goRight():
    ReleaseKey(LEFT)
    PressKey(RIGHT)

#def startShooting():
#    PressKey(SPACE)

#def stopShooting():
#    ReleaseKey(SPACE)

if __name__ == "__main__":
    startGame()