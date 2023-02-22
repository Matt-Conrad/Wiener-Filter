from generation import generateBGs
from diffusion import applyDiffusion
from analysis import analyze

def main():
    # bg = generateBGs(1, 40.0) # Time in hours
    # bg.plot.line(subplots=False)

    # bg.to_pickle("150Patients47hr.pkl") 

    import pandas as pd
    bg = pd.read_pickle("150Patients47hr.pkl")

    x = applyDiffusion(bg.iloc[3:], 1)

    analyze(x)

    print("done")

if __name__ == "__main__":
    main()