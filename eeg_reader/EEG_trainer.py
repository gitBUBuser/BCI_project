import time
import EEG
import threading

movements = ["Default", "Right hand right", "Left hand left"]
recorder = EEG.EEG_recorder("COM5")
target = threading.Thread(target=recorder.record(), args=(1,))
for x in range(5):
    
    for i in range(len(movements)):
        print(movements[i])
        time.sleep(1)
        begin = time.time()


        target.start()

        while (time.time()-begin < 8):
            pass
        target._stop()


        time.sleep(0.1)
        recorder.save_file(movements[i] + " " + str(x) + ".wav")
        print("Stop")
        time.sleep(0.5)