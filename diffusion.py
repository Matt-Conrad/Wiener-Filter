from scipy import signal, fft, fftpack
from noise import calculateNoise, calculateSNR
from wiener import calculateWienerDirect, calculateWienerIterative, applyWiener
from plotting import plotOptimalWiener

import numpy as np
import matplotlib.pyplot as plt

b = [0.0952, 0]
a = [1, -0.9048]

targetSNR = 40 # dB

filterOrder = 30

def applyDiffusion(df, nDatasets):
    df = df.iloc[:, 0:nDatasets]

    S = df.copy()
    Y = df.copy()

    for i, bg in enumerate(df):
        s = df[bg].to_numpy()

        y = signal.lfilter(b, a, s)

        S[bg] = s
        Y[bg] = y

    # Remove transitory effect introduced by filtering
    cropIndex = 300
    S = S.iloc[cropIndex:, 0:nDatasets]
    Y = Y.iloc[cropIndex:, 0:nDatasets]

    return S, Y

def applyNoise(Y):
    X = Y.copy()

    for i, bg in enumerate(Y):
        y = Y[bg].to_numpy()

        noise = calculateNoise(y, 15)

        x = y + noise

        SNR = calculateSNR(x)

        X[bg] = x

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

