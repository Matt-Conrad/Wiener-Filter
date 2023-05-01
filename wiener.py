import scipy.optimize as opt
import numpy as np

Ts = 1 # minute
Ts = Ts / 60 # 60 minutes = 1 hour

filterOrder = 30

def calculateWienerDirect(x, S, p):
    s = S[x.name]

    samplePeriod = 20

    n = int((len(x) - p) / samplePeriod)

    Y = np.zeros((n, p))
    s_vec = np.zeros((n, 1))

    for i in range(n):

        firstSample = (i * samplePeriod) + p
        lastSample = i * samplePeriod

        section = x[lastSample:firstSample]

        Y[i, :] = np.flip(section)

        s_vec[i] = s[firstSample]

    firstTerm = np.matmul(np.transpose(Y), Y)

    secondTerm = np.linalg.inv(firstTerm)

    thirdTerm = np.matmul(secondTerm, np.transpose(Y))

    g_opt = np.matmul(thirdTerm, s_vec)

    return g_opt[:, 0]

def calculateWienerIterative(x, p):
    samplePeriod = 20

    n = int((len(x) - p) / samplePeriod)

    Y = np.zeros((n, p))
    s_vec = np.zeros((n, 1))

    def residualFunc(g):
        sPrime = np.zeros(s_vec.shape)
        
        for i in range(0, Y.shape[0]):
            for j in range(0, Y.shape[1]):
                sPrime[i] += g[j] * Y[i, j]
            
        residuals = sPrime - s_vec

        return residuals[:, 0]

    f = residualFunc
    x0 = np.ones((p,))
    method = "lm"

    g_opt = opt.least_squares(f, x0, method=method)

    return g_opt.x

def calculateWiener(S, X, directMethod=True):
    g_opt = None

    if directMethod:
        g_opt = X.apply(lambda x: calculateWienerDirect(x, S, filterOrder))
    else:
        g_opt = X.apply(lambda x: calculateWienerIterative(x, filterOrder))

    return g_opt
