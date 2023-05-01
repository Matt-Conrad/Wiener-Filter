from generation import generateBGs
from filtering import applyDiffusion, applyConvolution
from analysis import analyze
from plotting import plotOptimalWiener, plotSignals, plotMagnitudes
from wiener import calculateWiener
from noise import applyNoise

def main():
    # bg = generateBGs(2, 40.0) # Time in hours
    # bg.plot.line(subplots=False)

    # bg.to_pickle("data/pickle2") 

    import pandas as pd
    bg = pd.read_pickle("data/pickle2")

    S, Y = applyDiffusion(bg.iloc[3:], 2)

    X = applyNoise(Y, 15)

    g_opt = calculateWiener(S, X, directMethod=True)

    plotOptimalWiener(g_opt)

    bg = pd.read_pickle("data/testPickle")

    S, Y = applyDiffusion(bg.iloc[3:], 1)

    X = applyNoise(Y, 35)

    SPrime = applyConvolution(X, g_opt["BG1"].values)

    plotSignals(S, X, SPrime)

    plotMagnitudes(S, X, SPrime)

    analyze(X["BG0"].values)

    print("done")

if __name__ == "__main__":
    main()