import numpy as np
from scipy import fft

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

def calculateSNR(x):
    xMag = np.abs(fft.fft(x.values))
    xPSD = xMag ** 2

    noiseAvg = np.mean(xPSD[500:1400])
    noiseSum = noiseAvg * 1900

    signalSum = np.sum(xPSD) - noiseSum

    SNRestimate = 10 * np.log10(signalSum / noiseSum)

    return SNRestimate

def applyNoise(Y, snr):
    Noise = Y.apply(lambda y: calculateNoise(y, snr), axis=0)

    X = Y.add(Noise, fill_value=0)

    SNRs = X.apply(calculateSNR, axis=0)

    return X