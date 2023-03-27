from generation import generateBGs
from diffusion import applyDiffusion, applyNoise, applyWiener2
from analysis import analyze

def main():
    # bg = generateBGs(1, 40.0) # Time in hours
    # bg.plot.line(subplots=False)

    # bg.to_pickle("data/pickle") 

    import pandas as pd
    bg = pd.read_pickle("data/pickle")

    S, Y = applyDiffusion(bg.iloc[3:], 1)

    X = applyNoise(Y)

    x = applyWiener2(S, Y, X, directMethod=True)

    analyze(x)

    print("done")

if __name__ == "__main__":
    main()