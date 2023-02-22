from scipy import signal, fft, fftpack
import matplotlib.pyplot as plt
from generation import generateBGs
import pandas as pd
import scipy.optimize as opt

import numpy as np

Ts = 1 # minute
Ts = Ts / 60 # 60 minutes = 1 hour

targetSNR = 40 # dB

b = [0.0952, 0]
a = [1, -0.9048]

filterOrder = 30

def applyDiffusion(df, nDatasets):
    df = df.iloc[:, 0:nDatasets]

    w = 512
    g_opts = np.zeros((nDatasets, filterOrder))
    g_opt_mag = np.zeros((nDatasets, filterOrder))
    g_opt_ang = np.zeros((nDatasets, w))
    g_opt2 = None
    for i, bg in enumerate(df):
        s = df[bg].to_numpy()

        y = signal.lfilter(b, a, s)

        # Extract flat part only 
        # TODO: Take out later
        # s = s[300:2200]
        # y = y[300:2200]

        # Remove transitory effect introduced by filtering
        s = s[300:]
        y = y[300:]

        sFlat = s[:1900]
        yFlat = y[:1900]

        noise = calculateNoise(y, 15)
        noiseFlat = calculateNoise(yFlat, targetSNR)

        x = y + noise
        xFlat = yFlat + noiseFlat

        # plotSignalCreation(s, y, x)

        # plotFilterDetails(b, a)

        # plotPSD(y, noise, x)

        # calculateSNR(yFlat, noiseFlat, xFlat)

        g_opt, g_opt2 = calculateWiener(s, x, y, filterOrder)

        g_opts[i, :] = g_opt[:, 0]

        gMag = np.abs(fft.fft(g_opt))
        g_opt_mag[i, :] = gMag[:, 0]

        coeffs = g_opt[:, 0]

        _, gd = signal.group_delay((coeffs, np.array([1])), w=w)
        g_opt_ang[i, :] = gd

        # plt.plot(gMag, '.')
        # plt.show()

        # plt.plot(gd, '.')
        # plt.show()

        print("done")

    # gd = np.mean(g_opt_ang, axis=0)

    # gSize = g_opt.size
    # freqs = fftpack.fftfreq(gSize, Ts)
    # idx = np.argsort(freqs)

    # _, axes = plt.subplots(2, 1)

    # axes[0].plot(freqs[idx][gSize//2:], gMag[idx][gSize//2:], ".", label="x")
    # axes[1].plot(gd, ".")

    # plt.suptitle(f"{nDatasets}Patients47hr")
    
    # plt.show()

    applyWiener(g_opts[0, :], g_opt2)

    return x

def applyWiener(g_opt, g_opt2):
    nProfiles = 20

    # df = generateBGs(1, 67.0) # Time in hours
    # df.to_pickle("Test67hr.pkl") 
    df = pd.read_pickle("Test67hr.pkl")

    delay = 10
    delaySpread = 2

    for i, bg in enumerate(df):
        s = df[bg].to_numpy()

        y = signal.lfilter(b, a, s)

        # Remove transitory effect introduced by filtering
        s = s[300:]
        y = y[300:]

        noise = calculateNoise(y, 35)

        x = y + noise

        sPrime = np.convolve(x, g_opt)

        sPrime2 = np.convolve(x, g_opt2)

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

        axes[0].plot(s, label="s")
        # axes[0].plot(x, label="x")
        axes[0].plot(sPrime, label="sp")

        axes[1].plot(s, label="s")
        axes[1].plot(x, label="x")
        # axes[1].plot(sPrime2, label="sp")
        axes[1].legend()

        plt.show()


        freqs = fftpack.fftfreq(s.size, Ts)
        idx = np.argsort(freqs)

        # gMag = np.abs(fft.fft(g_opt))

        # plt.semilogy(gMag)
        # plt.show()
        
        plt.semilogy(freqs[idx][s.size//2:], np.abs(fft.fft(s))[idx][s.size//2:], label="s")
        plt.semilogy(freqs[idx][x.size//2:], np.abs(fft.fft(x))[idx][x.size//2:], label="x")
        plt.semilogy(freqs[idx][sPrime.size//2:], np.abs(fft.fft(sPrime))[idx][sPrime.size//2:], label="sp")

        # plt.plot(freqs[idx][sPrime.size//2:], np.abs(fft.fft(sPrime))[idx][sPrime.size//2:], label="sp")

        plt.legend()

        plt.show()


def calculateWiener(s, x, y, p):
    Ts = 1 # minute
    Ts = Ts / 60 # 60 minutes = 1 hour
    samplePeriod = 20

    n = int((len(x) - p) / samplePeriod)

    Y = np.zeros((n, p))
    s_vec = np.zeros((n, 1))

    for i in range(n):

        firstSample = (i * samplePeriod) + p
        lastSample = i * samplePeriod

        section = x[lastSample:firstSample]

        Y[i, :] = np.flip(section)

        s_vec[i] = s[firstSample]

    firstTerm = np.matmul(np.transpose(Y), Y)

    secondTerm = np.linalg.inv(firstTerm)

    thirdTerm = np.matmul(secondTerm, np.transpose(Y))

    g_opt = np.matmul(thirdTerm, s_vec)


    ###

    def residualFunc(g):
        sPrime = np.zeros(s_vec.shape)
        
        for i in range(0, Y.shape[0]):
            for j in range(0, Y.shape[1]):
                sPrime[i] += g[j] * Y[i, j]
            
        residuals = sPrime - s_vec

        return residuals[:, 0]

    #fun
    #x0
    f = residualFunc
    # x0 = g_opt[:,0]
    x0 = np.ones(g_opt[:,0].shape)
    method = "lm"

    g_opt2 = opt.least_squares(f, x0, method=method)

    return g_opt, g_opt2.x



def calculateSNR(y, noise, x):
    # All viable ways to calculate SNR

    noiseMag = np.abs(fft.fft(noise))
    yMag = np.abs(fft.fft(y))

    noisePSD = noiseMag ** 2
    yPSD = yMag ** 2

    SNRestimate = 10 * np.log10(np.sum(yPSD) / np.sum(noisePSD))
    
    xMag = np.abs(fft.fft(x))
    xPSD = xMag ** 2

    xStart = xPSD[:100]
    xEnd = xPSD[1800:]

    noiseSection = xPSD[100:1800]

    signalSectionSum = np.sum(xStart) + np.sum(xEnd)

    SNRestimate2 = 10 * np.log10(signalSectionSum / np.sum(noiseSection))

    noiseAvg = np.mean(xPSD[500:1400])
    noiseSum = noiseAvg * 1900
    noiseSumBelow3 = noiseAvg * 200
    
    SNRestimate3 = 10 * np.log10((signalSectionSum - noiseSumBelow3) / noiseSum)

    signalSum = np.sum(xPSD) - noiseSum

    SNRestimate4 = 10 * np.log10(signalSum / noiseSum)

    print("Done")

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

def calculateNoise(y, targetSNR):
    yPow = y ** 2

    yAvgPow = np.mean(yPow)

    yAvgPowDB = 10 * np.log10(yAvgPow)

    noiseAvgPowDB = yAvgPowDB - targetSNR

    noiseAvgPow = 10 ** (noiseAvgPowDB / 10)

    meanNoise = 0

    # Average noise power == noise variance 
    noiseSTD = np.sqrt(noiseAvgPow)

    noise = np.random.normal(meanNoise, noiseSTD, len(y))

    var = np.var(noise)

    n2 = noise ** 2

    nAvgPow = np.mean(n2)

    nAvgPowDB = 10 * np.log10(nAvgPow)

    return noise

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