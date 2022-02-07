#!/usr/bin/python2.7
# Homework 1 Code
import numpy as np
import matplotlib.pyplot as plt

def find_misclassified(w, x, y):
    #returns index of misclassified data
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    for i in range(x.shape[0]):
        prediction = np.sign(np.sum(w*x[idxs[i],:]))
        if prediction != y[idxs[i]]:
            return idxs[i]
    return -1

def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Your code here, assign the proper values to w and iterations:
    max_iter = 10000000
    X = data_in[:,0:-1]
    Y = data_in[:,-1]
    w = np.zeros(X.shape[1])

    iterations = 0
    while iterations < max_iter:
        mis_idx = find_misclassified(w,X,Y)
        if mis_idx != -1:
            #update weights
            w_t1 = w + Y[mis_idx]*X[mis_idx,]
            w = w_t1
        else:
            #found optimal seperator!
            # x_plt = np.linspace(-1,1,1000)
            # y_plt = (-w[1]*x_plt-w[0])/w[2]
            # plt.plot(x_plt,y_plt,c='y',linestyle='dashed')
            # plt.show()
            return w,iterations
        iterations += 1
    return w, iterations

def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW1
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

    # Generate dataset
    x = np.random.uniform(low=-1,high=1,size=[N,d])
    x_0 = np.ones([N,1])
    X = np.concatenate((x_0,x),axis=1)

    w_star = np.random.uniform(low=0,high=1, size=d+1)
    w_star[0] = 0

    Y = np.zeros([N,1])
    for i in range(N):
        if np.sum(X[i,]*w_star) >= 0:
            Y[i] = 1
            # plt.scatter(X[i,1],X[i,2],c="green")
        else:
            Y[i] = -1
            # plt.scatter(X[i,1],X[i,2],c="red")

    # x = np.linspace(-1,1,1000)
    # y = (-w_star[1]*x-w_star[0])/w_star[2]
    # plt.plot(x,y,c="blue")
    # plt.xlim([-1,1])
    # plt.ylim([-1,1])

    # compute paramaters
    rho = float("inf")
    R = float("-inf")
    for i in range(N):
        if np.sum(Y[i]*(w_star*X[i,])) < rho:
            rho = np.sum(Y[i]*(w_star*X[i,]))
        if np.linalg.norm(X[i,],ord=2)>R:
            R = np.linalg.norm(X[i,],ord=2)

    tmax = np.ceil(R*R*np.linalg.norm(w_star,ord=2)*np.linalg.norm(w_star,ord=2)/rho/rho)
    num_iters = []
    bounds_minus_ni = []
    dataset = np.concatenate((X,Y),axis=1)
    for _ in range(num_exp):
        output = perceptron_learn(dataset)
        num_iters.append(output[1])
        bounds_minus_ni.append(tmax-output[1])

        # #testing if it worked
        # test_y = Y*0
        # for i in range(Y.shape[0]):
        #     test_y[i]=np.sign(np.sum(output[0]*X[i,]))
        # print(np.sum(Y-test_y))

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
