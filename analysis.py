import numpy as np

def analyze(x):

    xPow = x ** 2

    xAvgPow = np.mean(xPow)

    xAvgPowDB = 10 * np.log10(xAvgPow)

    _ = xAvgPowDB

    print("Done")
