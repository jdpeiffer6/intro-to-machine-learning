import pandas as pd
import numpy as np

def relabelData(data:pd.DataFrame):
    """Changes labels of 0 to -1 by reference"""
    for i in range(data.shape[0]):
        if data.iloc[i,-1] == 0:
            data.iloc[i,-1] = -1

def findGradient(w:np.ndarray, X:np.ndarray, y: np.ndarray):
    # w: weight vector of dim d+1
    # X: feature matrix of dim N, d+1
    # Y: classifications
    gradient = np.zeros(w.shape)
    N = y.size
    
    #more vectorized
    tops = np.multiply(X,y)
    bottoms = X.dot(w)
    bottoms = 1 + np.exp( np.multiply(bottoms,y.squeeze()) )
    gradient = np.divide(tops,bottoms[:,None])
    gradient = np.sum(gradient,axis=0) / (-N)

    return gradient

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def findCrossEntropyError(w,X,y):
    N = y.size
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis = 1)
    ce_error = 0
    wdotx = X.dot(w)
    for n in range(N):
        ce_error += np.log(1 + np.exp( -y[n] * wdotx[n] ))
    ce_error = ce_error/N

    return ce_error