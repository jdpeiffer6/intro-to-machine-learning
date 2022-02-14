# %%
import numpy as np
import matplotlib.pyplot as plt

# %% (b)
first = []
rand = []
min = []
runs = 100000
for _ in range(runs):
    c = np.random.binomial(n=10,p=0.5,size=1000)

    v1 = c[0]
    vrand = c[np.random.randint(0,999)]
    vmin = np.min(c)

    first.append(v1/10)
    rand.append(vrand/10)
    min.append(vmin/10)

plt.subplot(1,3,1)
plt.hist(first,color='r',alpha=0.5)
plt.xlim([0,1])
plt.title(r'$\nu_1$')
plt.xlabel('Fraction of heads (10 total flips)')
plt.ylabel('Count')
plt.subplot(1,3,2)
plt.hist(rand,color='g',alpha=0.5)
plt.xlim([0,1])
plt.title(r'$\nu_{rand}$')
plt.xlabel('Fraction of heads (10 total flips)')
plt.subplot(1,3,3)
plt.hist(min,color='b',alpha=0.5)
plt.xlim([0,1])
plt.title(r'$\nu_{min}$')
plt.xlabel('Fraction of heads (10 total flips)')
plt.suptitle('1000 coins flipped 10 times\nExperiement ran 100,000 times')
plt.show()

# %% (c)
def hoeffdingBound(ep,n):
    return 2.0*np.exp(-2.0*n*ep**2)
ep = np.arange(0,0.5,0.01)
hb = hoeffdingBound(ep,10)
v1 = np.abs(np.asarray(first)-0.5)
vrand = np.abs(np.asarray(rand)-0.5)
vmin = np.abs(np.asarray(min)-0.5)
p1_d = np.zeros([ep.size])
min_d = np.zeros([ep.size])
rand_d = np.zeros([ep.size])

for i in range(ep.size):
    epsilion = ep[i]
    p1_d[i] = np.sum(v1 > epsilion)/runs
    min_d[i] = np.sum(vmin> epsilion)/runs
    rand_d[i] = np.sum(vrand > epsilion)/runs

plt.plot(ep,hb,'b')
plt.plot(ep,p1_d,'y--')
plt.plot(ep,min_d,'g--')
plt.plot(ep,rand_d,'r--')
plt.legend(['Hoeffding Bound','First Coin','Minimum Coin','Random Coin'])
plt.xlabel(r'$\epsilon$')
plt.show()

# %%
