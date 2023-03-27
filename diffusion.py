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

def applyDiffusion2(df, nDatasets, directMethod=True):
    Ts = 1 # minute
    Ts = Ts / 60 # 60 minutes = 1 hour

    df = df.iloc[:, 0:nDatasets]

    S = df.copy()
    Y = df.copy()

    for i, bg in enumerate(df):
        s = df[bg].to_numpy()

        y = signal.lfilter(b, a, s)

        S[bg] = s
        Y[bg] = y

    return S, Y

def applyDiffusion(S, Y, nDatasets, directMethod=True):
    Ts = 1 # minute
    Ts = Ts / 60 # 60 minutes = 1 hour

    # Remove transitory effect introduced by filtering
    cropIndex = 300
    S = S.iloc[cropIndex:, 0:nDatasets]
    Y = Y.iloc[cropIndex:, 0:nDatasets]

    g_opt = None

    for i, bg in enumerate(S):
        s = S[bg].to_numpy()

        y = Y[bg].to_numpy()

        noise = calculateNoise(y, 15)

        x = y + noise

        # plotSignalCreation(s, y, x)

        # plotFilterDetails(b, a)

        # plotPSD(y, noise, x)

        SNR = calculateSNR(x)

        if directMethod:
            g_opt = calculateWienerDirect(s, x, y, filterOrder)
        else:
            g_opt = calculateWienerIterative(x, filterOrder)
        
    plotOptimalWiener(g_opt)

    applyWiener(g_opt)

    return x

