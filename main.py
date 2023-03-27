from generation import generateBGs
from diffusion import applyDiffusion, applyDiffusion2
from analysis import analyze

def main():
    # bg = generateBGs(1, 40.0) # Time in hours
    # bg.plot.line(subplots=False)

    # bg.to_pickle("data/pickle") 

    import pandas as pd
    bg = pd.read_pickle("data/pickle")

    S, Y = applyDiffusion2(bg.iloc[3:], 1, directMethod=True)

    x = applyDiffusion(S, Y, 1, directMethod=True)

    analyze(x)

    print("done")

if __name__ == "__main__":
    main()