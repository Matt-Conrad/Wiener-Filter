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

def calculateSNR(y, noise, x):
    # All viable ways to calculate SNR

    noiseMag = np.abs(fft.fft(noise))
    yMag = np.abs(fft.fft(y))

    noisePSD = noiseMag ** 2
    yPSD = yMag ** 2

    SNRestimate = 10 * np.log10(np.sum(yPSD) / np.sum(noisePSD))
    
    xMag = np.abs(fft.fft(x))
    xPSD = xMag ** 2

    xStart = xPSD[:100]
    xEnd = xPSD[1800:]

    noiseSection = xPSD[100:1800]

    signalSectionSum = np.sum(xStart) + np.sum(xEnd)

    SNRestimate2 = 10 * np.log10(signalSectionSum / np.sum(noiseSection))

    noiseAvg = np.mean(xPSD[500:1400])
    noiseSum = noiseAvg * 1900
    noiseSumBelow3 = noiseAvg * 200
    
    SNRestimate3 = 10 * np.log10((signalSectionSum - noiseSumBelow3) / noiseSum)

    signalSum = np.sum(xPSD) - noiseSum

    SNRestimate4 = 10 * np.log10(signalSum / noiseSum)

    print("Done")
