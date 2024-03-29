from scipy import signal
import numpy as np

targetSNR = 40 # dB

def applyDiffusion(S, nDatasets):
    S = S.iloc[:, 0:nDatasets]

    b = [0.0952, 0]
    a = [1, -0.9048]

    Y = S.apply(lambda s: signal.lfilter(b, a, s), axis=0)

    # Remove transitory effect introduced by filtering
    cropIndex = 300
    S = S.iloc[cropIndex:, 0:nDatasets]
    Y = Y.iloc[cropIndex:, 0:nDatasets]

    return S, Y

def applyConvolution(X, g_opt):
    SPrime = X.apply(lambda x: np.convolve(x, g_opt), axis=0)

    return SPrime
