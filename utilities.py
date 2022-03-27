import pandas as pd
import numpy as np
import sys
from numpy.matlib import repmat
from collections import Counter

def gtm_rctg(samp_size):
    xDim = samp_size[0]
    yDim = samp_size[1]
    X,Y = np.meshgrid(np.linspace(0,xDim-1,xDim),np.linspace(yDim-1,0,xDim))
    X = np.concatenate(X.T.reshape(-1,1))
    Y = np.concatenate(Y.T.reshape(-1,1))
    sample = np.array([X,Y])
    maxXY = [np.max(X),np.max(Y)]
    sample[0] = 2 * (sample[0]-maxXY[0]/2)/maxXY[0]
    sample[1] = 2 * (sample[1]-maxXY[1]/2)/maxXY[1]
    return sample.T

def dist2(x,c):
    ndata,dimx = np.shape(x)
    ncentres,dimc = np.shape(c)
    if (dimx != dimc):
        print("Data dimension does not match dimension of centres")
    x_sq_T = np.transpose([a*a for a in (x)])
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
