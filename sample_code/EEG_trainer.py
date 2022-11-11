import time

movements = ["Default", "Right hand right", "Left hand left", "Right leg right", "Left leg left"]


for i in range (4):
    print(movements[i])
    time.sleep(2)
    begin = time.time()
    while (time.time()-begin < 8):
        #recording
        pass
    print("Stop")
    time.sleep(2)