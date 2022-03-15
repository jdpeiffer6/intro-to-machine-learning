#!/usr/bin/python3
# Homework 4 Code
from cgi import test
from re import L
from black import out
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def bagged_trees(X_train, y_train, X_test, y_test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees 
    # and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: You may use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function

    # bag data and train trees
    trees:List[DecisionTreeClassifier] = []
    bootstrap_log=np.zeros((y_train.size,num_bags))
    N = y_train.size
    for m in range(num_bags):
        #create dataset
        indexes = np.random.choice(N,N)
        X_bag = X_train[indexes,:]
        y_bag = y_train[indexes]
        bootstrap_log[indexes,m]=1

        #train individual tree
        tree = DecisionTreeClassifier()
        tree.fit(X_bag,y_bag)
        trees.append(tree)    #add to forest
        

    # #this is kinda cool
    # plt.imshow(bootstrap_log)
    # plt.show()

    # calculate OOB error
    out_of_bag_error = 0
    for n in range(N):
        oob = np.where(bootstrap_log[n,:]==0)[0]
        x_n = X_train[n,:]
        x_n = x_n.reshape(1,-1)
        predicted=0
        for m in oob:
            predicted += trees[m].predict(x_n)
        h_n = np.sign(predicted)
        out_of_bag_error += np.abs(h_n - y_train[n])
    out_of_bag_error/=N
    out_of_bag_error=out_of_bag_error[0]

    # caculate test error
    test_error = 0
    for n in range(y_test.size):
        predicted = 0
        x_n = X_test[n,:]
        x_n = x_n.reshape(1,-1)
        for tree in trees:
            predicted += tree.predict(x_n)
        h_n = np.sign(predicted)
        test_error += np.abs(h_n - y_test[n])
    test_error /= y_test.size
    test_error = test_error[0]

    return out_of_bag_error, test_error

    """
    -JD
    -Julia
    -John
    -Mary
    -Emily
    -Brett
    -Sydney"""

def main_hw4():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    #subset data
    #TODO: Add the other dataset
    test_data = og_test_data[np.where((og_test_data[:,0]==1) | (og_test_data[:,0]==3))[0],:]
    train_data = og_train_data[np.where((og_train_data[:,0]==1) | (og_train_data[:,0]==3))[0],:]
    
    num_bags = 100

    # Split data
    X_train = train_data[:,1:]
    y_train = train_data[:,0]
    y_train[np.where(y_train==3)]=int(-1)
    y_train[np.where(y_train==1)]=int(1)

    X_test = test_data[:,1:]
    y_test = test_data[:,0]
    y_test[np.where(y_test==3)]=int(-1)
    y_test[np.where(y_test==1)]=int(1)

    # Run bagged trees
    oob_error = []
    test_e = []
    for num_bags in range(1,201):
        out_of_bag_error, test_error = bagged_trees(X_train, y_train, X_test, y_test, num_bags)
        oob_error.append(out_of_bag_error)
        test_e.append(test_error)
        print(num_bags/2)
    plt.plot(oob_error)
    plt.plot(test_e)
    plt.legend(["OOB Error","Test Error"])
    plt.show()
    # train_error, test_error = single_decision_tree(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main_hw4()

