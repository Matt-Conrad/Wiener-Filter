import matplotlib.pyplot as plt
from scipy import signal, fft, fftpack
import numpy as np

def plotSignalCreation(s, y, x):
    plt.plot(s, label='s(n)')
    # plt.plot(y, label='y(n)')
    plt.plot(x, label='x(n)')

    plt.title("Relationship between BG and Sensor Signal")
    plt.xlabel("Time (in min)")
    plt.ylabel("Glucose Level (in mg/dL)")
    plt.legend()

    plt.show()

def plotFilterDetails(b, a):
    fig = plt.figure()

    w, h = signal.freqz(b, a) # w is in rad/sample

    Ts = 1 # minute
    Ts = Ts / 60 # 60 minutes = 1 hour
    fs = 1 / Ts # cycle per hour

    f = fs * w / (2 * np.pi) 

    ax1 = fig.add_subplot(221)

    mag = abs(h)

    ax1.plot(f, mag)
    ax1.set_xlabel('Frequency [cycle/hr]')
    ax1.set_ylabel('Amplitude')

    angles = np.unwrap(np.angle(h))

    ax2 = fig.add_subplot(222)

    ax2.plot(f, angles)
    ax2.set_xlabel('Frequency [cycle/hr]')
    ax2.set_ylabel('Angle (radians)')

    wd, gd = signal.group_delay((b, a))

    fd = fs * wd / (2 * np.pi) 

    ax3 = fig.add_subplot(223)

    ax3.plot(fd, gd)
    ax3.set_xlabel('Frequency [cycle/hr]')
    ax3.set_ylabel('Group Delay (samples)')

    ax4 = fig.add_subplot(224)

    ps = mag ** 2

    ax4.plot(f, ps)
    ax4.set_xlabel('Frequency [cycle/hr]')
    ax4.set_ylabel('Power')

    plt.show()

def plotPSD(y, noise, x):
    fig = plt.figure()

    yMag = np.abs(fft.fft(y))
    xMag = np.abs(fft.fft(x))
    noiseMag = np.abs(fft.fft(noise))

    ax1 = fig.add_subplot(311)
    ax1.semilogy(noiseMag, label="noise")
    ax1.semilogy(xMag, label="x")
    ax1.semilogy(yMag, label="y")
    ax1.legend()

    yPSD = yMag ** 2
    xPSD = xMag ** 2
    noisePSD = noiseMag ** 2

    Ts = 1 # minute
    Ts = Ts / 60 # 60 minutes = 1 hour
    
    N = y.size

    freqs = fftpack.fftfreq(yPSD.size, Ts)

    idx = np.argsort(freqs)

    ax2 = fig.add_subplot(312)
    ax2.set_title("Signal with Noise")
    ax2.set_xlabel("Frequency [cycle/hr]")
    ax2.set_ylabel("Power [dB]")
    ax2.semilogy(freqs[idx][N//2:], noisePSD[idx][N//2:], label="noise")
    ax2.semilogy(freqs[idx][N//2:], xPSD[idx][N//2:], label="x")
    ax2.legend()

    Rnn = np.correlate(noise, noise, mode='full')
    Rnn = Rnn[Rnn.size//2:]
    Snn = np.abs(fft.fft(Rnn))

    SnnAvg = np.ones(Snn.shape) * np.mean(Snn)
    noisePSDAvg = np.ones(Snn.shape) * np.mean(noisePSD)

    # Flips the xPSD so that 0 freq is in the middle.
    # This averages the 300 most negative frequencies
    xPSD2 = xPSD[idx]
    xPSDAvg = np.ones(Snn.shape) * np.mean(xPSD2[:300])

    ax3 = fig.add_subplot(313)
    ax3.semilogy(freqs[idx][N//2:], Snn[N//2:], label="Noise Autocorr")
    ax3.semilogy(freqs[idx][N//2:], SnnAvg[N//2:], label="Noise Avg Autocorr")
    ax3.semilogy(freqs[idx][N//2:], noisePSDAvg[N//2:], label="Avg Noise")
    ax3.semilogy(freqs[idx][N//2:], xPSDAvg[N//2:], label="Avg of X's high freq")
    ax3.legend()

    plt.show()

    print("Done")

def plotOptimalWiener(coeffs):
    Ts = 1 # minute
    Ts = Ts / 60 # 60 minutes = 1 hour
    w = 512

    gMag = np.abs(fft.fft(coeffs))

    _, gd = signal.group_delay((coeffs[:,0], np.array([1])), w=w)

    gSize = coeffs.size
    freqs = fftpack.fftfreq(gSize, Ts)
    idx = np.argsort(freqs)

    _, axes = plt.subplots(2, 1)

    axes[0].plot(freqs[idx][gSize//2:], gMag[idx][gSize//2:], ".", label="x")
    axes[1].plot(gd, ".")
    
    plt.show()
