import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
from scipy.io import wavfile
import pathlib
import numpy as np


fs_rate, data = wavfile.read("data/BYB_Recording_2022-10-14_10.12.46.wav")

data = np.array(data)

print("sampling rate: ")
print(fs_rate)

l_signal = len(data.shape)
print("channels: ")
print(l_signal)


N = data.shape[0]
print("Complete Samplings:", N)

seconds = N / float(fs_rate)
print("Seconds:", seconds)

Ts = 1.0 / fs_rate
print("Sampling rate in seconds -- timestep between samples: ", Ts)

t = scipy.arange(0, seconds, Ts) #creates a step vector-- from 0 to seconds (97)
# with stepsize Ts (1*10⁻⁴)


"""this is a test -- ignore if not needed"""
"""data_ds = scipy.signal.resample(data, int(500 * seconds))
print("downsampled data: ")
print(data_ds)
print(data_ds.shape)

plt.set_cmap('viridis')
plt.specgram(data_ds, NFFT=256, Fs=500, noverlap=250)
plt.ylim(0,90)
plt.show()

FFT_ds = abs(scipy.fft.fft(data_ds))
t_ds = scipy.arange(0, seconds, 500)
freqs = fftpack.fftfreq(dada_ds.size, t_ds)"""


"""not sure why i need to do all of this"""
average_step = 1000 

average_data = []
avg_fs_rate = fs_rate / average_step
average_Ts = 1.0 / avg_fs_rate

for i in range(0,len(data), average_step):
    if i + average_step <= len(data):
        average = 0
        for j in range(i, i + average_step):
            average += data[j]
        average /= average_step
        average_data.append(average)
    else:
        continue

t_average = scipy.arange(0, int(seconds / average_Ts) * average_Ts, average_Ts)





# one sided FFT range?

FFT = abs(scipy.fft.fft(data))
FFT_side = FFT[range(int(N/2))]

freqs = scipy.fftpack.fftfreq(data.size, t[1]-t[0])
fft_freqs = np.array(freqs)
freqs_side = freqs[range(int(N/2))] # one side frequency range
fft_freqs_side = np.array(freqs_side)




plt.subplot(311)
p = plt.plot(t_average, average_data, "g")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.subplot(312)
p1 = plt.plot(t, data, "g") # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(313)
p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count dbl-sided')
#plt.subplot(314)
#p3 = plt.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
#plt.xlabel('Frequency (Hz)')
#plt.ylabel('Count single-sided')
plt.show()



print(FFT)