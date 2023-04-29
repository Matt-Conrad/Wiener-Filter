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

    if directMethod:
        g_opt = X.apply(lambda x: calculateWienerDirect(x, S, filterOrder))
    else:
        g_opt = X.apply(lambda x: calculateWienerIterative(x, filterOrder))
        
    plotOptimalWiener(g_opt["BG"].values)

    applyWiener(g_opt["BG"].values)

    return X["BG"]

