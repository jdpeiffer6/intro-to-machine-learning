import numpy as np
import matplotlib.pyplot as plt

first = []
rand = []
min = []

for _ in range(100000):
    c = np.random.binomial(n=10,p=0.5,size=1000)

    v1 = c[0]
    vrand = c[np.random.randint(0,999)]
    vmin = np.min(c)

    first.append(v1)
    rand.append(vrand)
    min.append(vmin)

plt.hist(first,color='r',alpha=0.5)
plt.hist(rand,color='g',alpha=0.5)
plt.hist(min,color='b',alpha=0.5)
plt.legend(['First','Random','Minimum'])
plt.show()