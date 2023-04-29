from scipy import signal, fft, fftpack
from noise import calculateNoise, calculateSNR
from wiener import calculateWienerDirect, calculateWienerIterative, applyWiener
from plotting import plotOptimalWiener

import numpy as np
import matplotlib.pyplot as plt

targetSNR = 40 # dB

filterOrder = 30

b = [0.0952, 0]
a = [1, -0.9048]

def applyFilter(s):
    b = [0.0952, 0]
    a = [1, -0.9048]

    y = signal.lfilter(b, a, s)

    return y

def applyDiffusion(S, nDatasets):
    S = S.iloc[:, 0:nDatasets]

    Y = S.apply(applyFilter, axis=0)

    # Remove transitory effect introduced by filtering
    cropIndex = 300
    S = S.iloc[cropIndex:, 0:nDatasets]
    Y = Y.iloc[cropIndex:, 0:nDatasets]

    return S, Y

def calculateNoises(y):
    snr = 15
    noise = calculateNoise(y, snr)
    return noise

def applyNoise(Y):
    Noise = Y.apply(calculateNoises, axis=0)
    X = Y.add(Noise, fill_value=0)
    SNRs = X.apply(calculateSNR, axis=0)

    return X

def applyWiener2(S, Y, X, directMethod=True):
    g_opt = None

    for i, bg in enumerate(S):
        # plotSignalCreation(s, y, x)

        # plotFilterDetails(b, a)

        # plotPSD(y, noise, x)

        s = S[bg].to_numpy()
        y = Y[bg].to_numpy()
        x = X[bg].to_numpy()

        if directMethod:
            g_opt = calculateWienerDirect(s, x, y, filterOrder)
        else:
            g_opt = calculateWienerIterative(x, filterOrder)
        
    plotOptimalWiener(g_opt)

    applyWiener(g_opt)

    return x

