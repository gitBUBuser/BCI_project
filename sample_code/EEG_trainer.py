import time

movements = ["Default", "Right hand right", "Left hand left", "Right leg right", "Left leg left"]

for i in range (4):
    printed = False
    begin = time.time()
    while (time.time()-begin < 10):
        if (not printed):
            print(movements[i])
            printed = True