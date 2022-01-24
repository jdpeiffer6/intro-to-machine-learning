#!/usr/bin/python2.7
# Homework 0 Code
import numpy as np
import matplotlib.pyplot as plt


def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Your code here, assign the proper values to w and iterations:

    return w, iterations


def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW0
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)

    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))

    N=100
    d=2
    # Generate dataset
    x = np.random.rand(N,d)
    x_0 = np.ones([N,1])
    X = np.concatenate((x_0,x),axis=1)

    w_star = np.random.uniform(low=-1,high=1, size=d+1)

    Y = np.zeros(N)
    for i in range(N):
        if np.sum(X[i,]*w_star) >= 0:
            Y[i] = 1
            plt.scatter(X[:,1],X[:,2],c="green")
        else:
            Y[i] = -1
            plt.scatter(X[:,1],X[:,2],c="red")

 
    x = np.linspace(0,1,100)
    y = (-w_star[1]*x-w_star[0])/w_star[2]
    plt.plot(x,y)
    plt.show()
    # Your code here, assign the values to num_iters and bounds_minus_ni:

    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()
