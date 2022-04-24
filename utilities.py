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
    sum_x = np.array([sum(a) for a in zip(*x_sq_T)])
    ones = np.ones(ncentres,)
    x2 = np.zeros((ndata,ndata))
    for i in range (ncentres):
        for j in range (ndata):
            x2[i][j] = ones[i] * sum_x[j]
    # x2 =np.dot(np.ones(ncentres,), sum_x)

    c_sq_T = np.transpose([a*a for a in (c)])
    sum_c = np.array([sum(a) for a in zip(*c_sq_T)])
    # c2 = np.transpose(np.ones(ndata,) * sum_c)
    c2 = np.zeros((ndata, ncentres))
    for i in range (ndata):
        for j in range (ncentres):
            c2[i][j] = ones[i] * sum_c[j]

    temp = 2 * np.dot(x, np.transpose(c))
    n2 = x2.T+c2-temp
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


def inverselink(dist_type, x):
    if dist_type == 'bernoulli':
        y = 1 / (1 + np.exp(-x))
    elif dist_type == 'multinomial':
        x = x.T
        n = x.shape[0]
        x = x - np.repeat(np.max(x), n).reshape((-1, 1))
        x = np.exp(x)
        sort_sum = np.sum(np.sort(x))
        y = x / np.repeat(sort_sum, n).reshape((-1, 1))
        y = y.T
    return y
