import time
import EEG

movements = ["Default", "Right hand right", "Left hand left"]
recorder = EEG.EEG_recorder("/port/")
for x in range(5):
    for i in range(len(movements)):
        print(movements[i])
        time.sleep(1)
        begin = time.time()
        while (time.time()-begin < 8):
            recorder.record()
        time.sleep(0.1)
        recorder.save_file(movements[i] + " " + str(x))
        print("Stop")
        time.sleep(0.5)