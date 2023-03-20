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
    Ts = 1 # minute
    Ts = Ts / 60 # 60 minutes = 1 hour

    df = df.iloc[:, 0:nDatasets]

    w = 512
    g_opts = np.zeros((nDatasets, filterOrder))
    g_opt_mag = np.zeros((nDatasets, filterOrder))
    g_opt_ang = np.zeros((nDatasets, w))
    g_opt2 = None

    for i, bg in enumerate(df):
        s = df[bg].to_numpy()

        y = signal.lfilter(b, a, s)

        # Remove transitory effect introduced by filtering
        s = s[300:]
        y = y[300:]

        noise = calculateNoise(y, 15)

        x = y + noise

        # plotSignalCreation(s, y, x)

        # plotFilterDetails(b, a)

        # plotPSD(y, noise, x)

        SNR = calculateSNR(x)

        g_opt = calculateWienerDirect(s, x, y, filterOrder)

        g_opt2 = calculateWienerIterative(x, filterOrder)

    plotOptimalWiener(g_opt)

    applyWiener(g_opt[:, 0], g_opt2)

    return x

