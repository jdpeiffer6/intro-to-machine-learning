def mH(n : int):
    return 1 + n + n*(n-1)*(n-2)/6

for i in range(50):
    print("%d:\t%d\t%.1f\n"%(i,2**i,mH(i)))