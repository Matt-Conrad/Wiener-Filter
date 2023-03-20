from scipy import signal, fft
from noise import calculateNoise, calculateSNR
from wiener import calculateWienerDirect, calculateWienerIterative, applyWiener

import numpy as np

b = [0.0952, 0]
a = [1, -0.9048]

targetSNR = 40 # dB

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

        g_opts[i, :] = g_opt[:, 0]

        gMag = np.abs(fft.fft(g_opt))
        g_opt_mag[i, :] = gMag[:, 0]

        coeffs = g_opt[:, 0]

        _, gd = signal.group_delay((coeffs, np.array([1])), w=w)
        g_opt_ang[i, :] = gd

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

