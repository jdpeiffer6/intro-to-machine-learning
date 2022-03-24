#!/usr/bin/python3
# Homework 4 Code
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from time import time

def bagged_trees(X_train, y_train, X_test, y_test, num_bags,trees,bootstrap_log):
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

    # #this is kinda cool
    # plt.imshow(bootstrap_log)
    # plt.show()

    #calculate OOB error try
    votes = np.zeros(bootstrap_log.shape,dtype=int)
    s=np.zeros(y_train.size,dtype=int)
    for m,tree in enumerate(trees):
        oob = np.where(~bootstrap_log[:,m] +2)[0]  # make sure this works. ok it does
        votes[oob,m] = tree.predict(X_train[oob,:])
        s[oob] += 1
    s = np.where(s!=0)[0]
    y_oob = np.sign(np.sum(votes,axis=1)) 
    out_of_bag_error = np.sum(~np.equal(y_oob[s],y_train[s]))/s.size

    # caculate test error
    votes = np.zeros(y_test.size,dtype=int)
    for m,tree in enumerate(trees):
        votes=np.add(tree.predict(X_test),votes)
    votes = np.sign(votes)
    test_error = np.sum(~np.equal(votes,y_test))/y_test.size
    return out_of_bag_error, test_error

def single_decision_tree(X_train, y_train, X_test, y_test):
    tree = DecisionTreeClassifier(criterion='entropy')
    tree.fit(X_train,y_train)
    train_predicted = tree.predict(X_train)
    test_predicted = tree.predict(X_test)

    train_error = np.sum(~np.equal(y_train,train_predicted))/np.size(train_predicted)
    test_error = np.sum(~np.equal(y_test,test_predicted))/np.size(test_predicted)
    
    return train_error, test_error
    

def main_hw4():
    t1 = time()
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    #subset data
    # 1 vs 3 or 3 vs 5
    num1 = 1
    num2 = 3
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

    # create bootstrapped dataset
    trees:List[DecisionTreeClassifier] = []
    bootstrap_log=np.zeros((y_train.size,200),dtype=int)
    N = y_train.size
    for m in range(200):
        #create dataset
        indexes = np.random.choice(N,N)   #is it ok that not every point here is unique?
        # indexes = np.unique(indexes)
        X_bag = X_train[indexes,:]
        y_bag = y_train[indexes]
        bootstrap_log[indexes,m] = 1

        #train individual tree
        tree = DecisionTreeClassifier(criterion='entropy')
        tree.fit(X_bag,y_bag)
        trees.append(tree)    #add to forest

    # plt.imshow(bootstrap_log)
    # plt.show()
    # Run bagged trees
    oob_error = []
    test_e = []
    num_bags = 200
    for n_bags in range(1,num_bags+1):
        out_of_bag_error, test_error = bagged_trees(X_train, y_train, X_test, y_test, n_bags, trees[:n_bags],bootstrap_log[:,:n_bags])
        oob_error.append(out_of_bag_error)
        test_e.append(test_error)
        if n_bags % 20 == 0:
            print(n_bags/num_bags) 
    plt.plot(np.arange(1,num_bags+1,1),oob_error)
    plt.plot(np.arange(1,num_bags+1,1),test_e)
    plt.legend(["OOB Error","Test Error"])
    train_error, test_error = single_decision_tree(X_train, y_train, X_test, y_test)
    print("Single train, test error: %.3f,%.3f"%(train_error,test_error))
    t2 = time()
    print("Execution Time: %.0f s"%(t2-t1))
    plt.xlabel("# of Bags")
    plt.ylabel("Out of Bag Error")
    plt.show()

if __name__ == "__main__":
    main_hw4()

