import time
import EEG
import threading
import multiprocessing



movements = ["Default", "Right hand right", "Left hand left"]
recorder = EEG.EEG_recorder("/dev/ttyACM")

def update_recorder():
    while(True):
        recorder.update()


update = multiprocessing.Process(target=update_recorder)
update.start()

def main():
    for x in range(5):
        for i in range(len(movements)):
            print(movements[i])
            time.sleep(0.5)

            begin = time.time()
            recorder.start_recording()
            time.sleep(5)
            recorder.save_file(movements[i] + " " + str(x))

            time.sleep(0.1)
            print("Stop")
            time.sleep(0.5)

    update.join()
    

if __name__ == "__main__":
    main()

