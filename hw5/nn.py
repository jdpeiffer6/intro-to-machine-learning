import numpy as np

def L1(x,w1,w2):
    # w1 = np.array((-2.5,1,1,-1))
    # w2 = np.array((-1.5,-1,-1,1))
    # print(np.sign(x.dot(w1)))
    # print(np.sign(x.dot(w2)))
    return np.array([1,np.sign(x.dot(w1)),np.sign(x.dot(w2))])

def L2(x,w3):
    # w3 = np.array((0.5,1,1))
    return np.sign(x.dot(w3))

def nn(x,w1,w2,w3):
    return L2(L1(x,w1,w2),w3)

if __name__ == "__main__":
    X = [np.array([1,1,1,1]),np.array([1,1,1,-1]),np.array([1,1,-1,1]),np.array([1,-1,1,1]),
    np.array([1,1,-1,-1]),np.array([1,-1,-1,1]),np.array([1,-1,1,-1]),np.array([1,-1,-1,-1])]
    Y = [-1,1,1,1,-1,1,-1,-1]
    b_choices = np.array([-2.5,-1.5,-0.5,0.5,1.5,2.5])
    w_choices = np.array([-1,1,-2,2])
    for _ in range(1000000):
        # w1 = np.array([np.random.choice(b_choices),np.random.choice(w_choices),np.random.choice(w_choices),np.random.choice(w_choices)])
        w1 = np.array([-2.5,1,1,-1])
        w2 = np.array([np.random.choice(b_choices),np.random.choice(w_choices),np.random.choice(w_choices),np.random.choice(w_choices)])
        # w3 = np.array([np.random.choice(b_choices),np.random.choice(w_choices),np.random.choice(w_choices)])
        w3 = np.array([1.5,1,1])
        # w1 = np.array((-2.5,1,1,-1))
        # w2 = np.array((-1.5,-1,-1,1))
        # w3 = np.array((0.5,1,1))
        sum = 0
        for x,y in zip(X,Y):
            y_test = nn(x,w1,w2,w3)
            if y_test == y:
                sum += 1
            if sum == 8:
                print("Found")
                print(w1)
                print(w2)
                print(w3)
                break