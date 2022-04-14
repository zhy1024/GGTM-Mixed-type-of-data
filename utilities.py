import pandas as pd
import numpy as np
import sys
from numpy.matlib import repmat
from collections import Counter

def gtm_rctg(samp_size):
    xDim = samp_size[0]
    yDim = samp_size[1]
    X, Y = np.meshgrid(np.arange(xDim), np.flip(np.arange(yDim)))
    maxX = np.max(X)
    maxY = np.max(Y)
    sample = np.array(np.concatenate((X.T.flatten().reshape((-1, 1)), Y.T.flatten().reshape((-1, 1))), axis=1), dtype=np.float64)
    sample[:, 0] = 2 * (sample[:, 0] - maxX / 2) / maxX
    sample[:, 1] = 2 * (sample[:, 1] - maxY / 2) / maxY
    return sample

def dist2(x,c):
    ndata,dimx = np.shape(x)
    ncentres,dimc = np.shape(c)
    if (dimx != dimc):
        print("Data dimension does not match dimension of centres")

    # if dimx != 2:
    #     print("dimension of x is not 2!")

    x_sq_T = np.transpose([[x[i][j] ** 2 for j in range(len(x[i]))] for i in range(len(x))])
    sum_x = np.array([sum(a) for a in zip(*x_sq_T)]).reshape(-1,1)
    x2 = np.ones(ncentres,) * sum_x
    c_sq_T = np.transpose([a*a for a in (c)])
    sum_c = np.array([sum(a) for a in zip(*c_sq_T)]).reshape(-1,1)
    c2 = np.transpose(np.ones(ndata,) * sum_c)
    temp = 2 * np.dot(x, np.transpose(c))
    n2 = x2+c2-temp
    return n2

def OnetoNcoding(data):
    Neachfeatures = []
    Coded = []
    for i in range (data.shape[1]):
        OneFeature = [[d[i]] for d in data]
        tmpFeature = OneFeature
        # nan_ind = np.isnan(tmpFeature)
        # tmpFeature[nan_ind] = []
        UniqueOneFeature = np.sort(np.unique(tmpFeature))
        OneFeatureCode = np.zeros((data.shape[0],UniqueOneFeature.shape[0]))
        Neachfeatures.append(UniqueOneFeature.shape[0])
        for j in range (UniqueOneFeature.shape[0]):
            temp = np.array(OneFeature) == repmat(UniqueOneFeature[j],len(OneFeature),1)
            for b in range (len(temp)):
                if temp[b] == [True]:
                    OneFeatureCode[b][j] = 1

        # not consider NaN
        if i == 0:
            Coded = OneFeatureCode
        else:
            Coded = np.hstack((Coded, OneFeatureCode))

    return Coded, Neachfeatures
