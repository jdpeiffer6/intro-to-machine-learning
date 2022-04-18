#!/usr/bin/python3
# Homework 5 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using decision stumps as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use
    train_error = []
    test_error = []
    stumps = [] #stores all the weak learners
    alphas = [] #stores each learners weights
    D_t = np.repeat(1/y_train.size,y_train.size)    #initialize dataset weights
    for _ in range(n_trees):
        # train and append stump
        stump = DecisionTreeClassifier(criterion='entropy',max_depth=1)
        stump.fit(X_train,y_train,sample_weight=D_t)     #Trains the stump with the weighted dataset
        stumps.append(stump)

        #calculate error
        h_t = stump.predict(X_train)
        e_t = np.sum(D_t[h_t != y_train])    #calculates the epsilon term for this iteration

        #calculate alpha_t
        alpha_t = 0.5*np.log((1-e_t)/e_t)
        alphas.append(alpha_t)

        #update weights
        D_t = D_t * np.exp(-alpha_t * y_train * h_t)
        D_t = D_t/np.sum(D_t)

        #compute test and train errors
        y_train_pred = y_train*0
        y_test_pred = y_test*0
        for alpha,stump in zip(alphas,stumps):    #aggregates predictions
            y_train_pred = np.add(y_train_pred,alpha*stump.predict(X_train))
            y_test_pred = np.add(y_test_pred,alpha*stump.predict(X_test))
        y_train_pred = np.sign(y_train_pred)    
        y_test_pred = np.sign(y_test_pred)    
        train_error.append(np.sum(~np.equal(y_train_pred,y_train))/y_train.size)   #computes error
        test_error.append(np.sum(~np.equal(y_test_pred,y_test))/y_test.size)


    return train_error, test_error   #returns a list of errors, one for each t


def main_hw5():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    num_trees = 200

    numbers = ((1,3),(3,5))
    #the algorithm will run twice, once for 1,3 and once for 3,5
    for num in numbers:
    #subset data
        num1 = num[0]
        num2 = num[1]
        test_data = og_test_data[np.where((og_test_data[:,0]==num1) | (og_test_data[:,0]==num2))[0],:]
        train_data = og_train_data[np.where((og_train_data[:,0]==num1) | (og_train_data[:,0]==num2))[0],:]
        
        # Split data
        X_train = train_data[:,1:]
        y_train = train_data[:,0]
        y_train[np.where(y_train==num1)]=int(-1)
        y_train[np.where(y_train==num2)]=int(1)

        X_test = test_data[:,1:]
        y_test = test_data[:,0]
        y_test[np.where(y_test==num1)]=int(-1)
        y_test[np.where(y_test==num2)]=int(1)

        train_error, test_error = adaboost_trees(X_train, y_train, X_test, y_test, num_trees)
        plt.plot(np.arange(num_trees),train_error,color='b')
        plt.plot(np.arange(num_trees),test_error,color='r')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title(str(num))
        plt.legend(['Training','Testing'])
        plt.show()
        
if __name__ == "__main__":
    main_hw5()
