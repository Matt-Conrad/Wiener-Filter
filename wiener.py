import scipy.optimize as opt
import numpy as np
import pandas as pd
from scipy import signal, fft, fftpack
from noise import calculateNoise
import matplotlib.pyplot as plt
from generation import generateBGs

Ts = 1 # minute
Ts = Ts / 60 # 60 minutes = 1 hour

def applyFilter(s):
    b = [0.0952, 0]
    a = [1, -0.9048]

    y = signal.lfilter(b, a, s)

    return y

def calculateWienerDirect(x, S, p):
    s = S[x.name]

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

    return g_opt[:, 0]

def calculateWienerIterative(x, p):
    samplePeriod = 20

    n = int((len(x) - p) / samplePeriod)

    Y = np.zeros((n, p))
    s_vec = np.zeros((n, 1))

    def residualFunc(g):
        sPrime = np.zeros(s_vec.shape)
        
        for i in range(0, Y.shape[0]):
            for j in range(0, Y.shape[1]):
                sPrime[i] += g[j] * Y[i, j]
            
        residuals = sPrime - s_vec

        return residuals[:, 0]

    f = residualFunc
    x0 = np.ones((p,))
    method = "lm"

    g_opt = opt.least_squares(f, x0, method=method)

    return g_opt.x

def calculateSPrime(x, g_opt):
    sPrime = np.convolve(x, g_opt)
    return sPrime

def applyWiener(g_opt):
    nProfiles = 20

    # df = generateBGs(1, 67.0) # Time in hours
    # df.to_pickle("Test67hr.pkl") 

    delay = 10
    delaySpread = 2

    S = pd.read_pickle("data/testPickle")

    Y = S.apply(applyFilter, axis=0)

    S = S.iloc[300:]
    Y = Y.iloc[300:]

    Noise = Y.apply(lambda y: calculateNoise(y, 35), axis=0)

    X = Y.add(Noise, fill_value=0)

    SPrime = X.apply(lambda x: calculateSPrime(x, g_opt), axis=0)

    for i, name in enumerate(S):
        s = S[name].values
        x = X[name].values
        sPrime = SPrime[name].values

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

        axes[0].plot(s, label="s")
        # axes[0].plot(x, label="x")
        axes[0].plot(sPrime, label="sp")

        axes[1].plot(s, label="s")
        axes[1].plot(x, label="x")
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

        print("done")

    print("done")

    return x
