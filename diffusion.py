from scipy import signal, fft, fftpack
from noise import calculateNoise, calculateSNR
from wiener import calculateWienerDirect, calculateWienerIterative, applyFilter

import numpy as np
import matplotlib.pyplot as plt

targetSNR = 40 # dB

filterOrder = 30

def applyDiffusion(S, nDatasets):
    S = S.iloc[:, 0:nDatasets]

    Y = S.apply(applyFilter, axis=0)

    # Remove transitory effect introduced by filtering
    cropIndex = 300
    S = S.iloc[cropIndex:, 0:nDatasets]
    Y = Y.iloc[cropIndex:, 0:nDatasets]

    return S, Y

def applyNoise(Y, snr):
    Noise = Y.apply(lambda y: calculateNoise(y, snr), axis=0)
    X = Y.add(Noise, fill_value=0)
    SNRs = X.apply(calculateSNR, axis=0)

    return X

def calculateWiener(S, X, directMethod=True):
    g_opt = None

    if directMethod:
        g_opt = X.apply(lambda x: calculateWienerDirect(x, S, filterOrder))
    else:
        g_opt = X.apply(lambda x: calculateWienerIterative(x, filterOrder))

    return g_opt

