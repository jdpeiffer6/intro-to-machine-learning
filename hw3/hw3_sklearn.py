#!/usr/bin/python3
# Homework 3 Code
import numpy as np
import time
from sklearn.linear_model import LogisticRegression

def relabelData(data:np.ndarray):
    """Changes labels of 0 to -1 by reference"""
    for i in range(data.shape[0]):
        if data.iloc[i,-1] == 0:
            data.iloc[i,-1] = -1

def findGradient(w:np.ndarray, X:np.ndarray, y: np.ndarray) -> np.ndarray:
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

def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # Your code here, assign the proper value to binary_error:
    N = y.size
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis = 1)
    binary_error = 0
    for n in range(N):
        x_n = X[n,:]
        y_predict = np.sign( sigmoid(w.dot(x_n)) - 0.5 )
        if y_predict != y[n]:
            binary_error += 1
    binary_error = binary_error/N
    return binary_error

def L2_update(learning_rate:float,lambdac:float) -> np.ndarray:
    return  2*learning_rate*lambdac
def logistic_reg(X, y, w_init, max_its, eta, grad_threshold,lambdac,regularization):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions; 
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:

    # add column of 1s
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis = 1)
    w = w_init
    t = 0
    while t < max_its:
        grads = findGradient(w,X,y)
        if np.all(grads < grad_threshold):
            # exit gradient descent
            break

        # #update without regularization
        # w = w - eta * grads

        # update
        if regularization == "L2":
            w = (1-2*eta*lambdac)*w - eta * grads
        elif regularization == "L1":
            #update with L1 regularization
            wprime = w - eta * grads
            w = wprime - eta*lambdac*np.sign(w)
            for i in np.where(wprime != 0)[0]:
                if (np.sign(w[i]) - np.sign(wprime[i]))!=0:
                    w[i] = 0
        else:
            print("Not a valid regularization type")
            return
        t += 1
    e_in = findCrossEntropyError(w,X[:,1:],y)
    return t, w, e_in


def main():
    # Load data
    X_train, X_test, y_train, y_test = np.load("digits_preprocess.npy", allow_pickle=True)
    y_train[np.where(y_train==0)] = -1
    y_test[np.where(y_test==0)] = -1
    # y_train.shape = (y_train.size,1)
    # y_test.shape = (y_test.size,1)

    w_0 = np.zeros(X_train.shape[1]+1)

    # normalize
    norms = np.zeros([2,X_train.shape[1]])
    for i in range(X_train.shape[1]):
        norms[0,i] = np.mean(X_train[:,i])
        norms[1,i] = np.std(X_train[:,i])
        X_train[:,i] = X_train[:,i] - norms[0,i]
        X_test[:,i] = X_test[:,i] - norms[0,i]
        if norms[1,i] == 0:
            X_test[:,i] = 0
        else:
            X_train[:,i] = X_train[:,i] / norms[1,i]
            X_test[:,i] = X_test[:,i] / norms[1,i]

    # sklearn
    model = LogisticRegression(penalty='l2',tol=10**-6,C=1/0.01,max_iter=10**4)
    model.fit(X_train,y_train)
    print(model.coef_)
    a=5 
    

if __name__ == "__main__":
    main()
