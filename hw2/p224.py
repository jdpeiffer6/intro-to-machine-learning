# %%
import numpy as np
import matplotlib.pyplot as plt

# # %% to calcualte g_bar
# iter = 1000
# a_bar = 0
# b_bar = 0
# for _ in range(iter):
#     x = np.random.uniform(low=-1,high=1,size=(2))
#     y=x**2
#     a = x[0]+x[1]
#     # b = (x[0]*y[1]-x[1]*y[0])/(x[0]-x[1])
#     b = -1*x[0]*x[1]
#     a_bar += a
#     b_bar += b
#     xx = np.linspace(-1,1,100)
#     yy = a*xx + b
#     # plt.scatter(x,y)
#     # plt.plot(xx,yy)
#     # plt.show()
# a_bar /= iter
# b_bar /= iter
# print("g bar statistics:\na = %.5f\nb = %.5f"%(a_bar,b_bar))

# %%
num_data_samples = 1000
num_x_samples = 1000
def gd_func(x,x1,x2):
    # hypothesis that the learning algorithm finds
    a = x1+x2
    b = -x1 * x2
    return a*x+b

def avg_g(x, gd_func,samples, target):
    gd_funcs = []
    for i in range(samples):
        sample = np.random.uniform(-1,1,2)
        v = gd_func(x,sample[0],sample[1])
        gd_funcs.append(v)
    avg_gfunc_at_x = np.mean(gd_funcs)
    var_gfunc_at_x = np.var(gd_funcs)
    bias_gfunc_at_x = (avg_gfunc_at_x - target(x))**2
    return avg_gfunc_at_x,var_gfunc_at_x,bias_gfunc_at_x

def calculate_bias_var_eout(gd_func, target_func, num_data, num_x):
    vars,biases,eouts = [], [], []
    for i in range(num_x):
        x = np.random.uniform(-1,1,1)
        _, variance, bias = avg_g(x, gd_func, num_data, target_func)
        vars.append(variance)
        biases.append(bias)
    
        #eout
        eout_on_data = [ ]
        for i in range(num_data_samples):
            sample = np.random.uniform(-1,1,2)
            v = gd_func(x,sample[0],sample[1])
            eout_on_data.append((v-target_func(x))**2)
        eout_data_avg = np.mean(eout_on_data)
        eouts.append(eout_data_avg)

    variance = np.mean(vars)
    bias = np.mean(biases)
    eout = np.mean(eouts)
    print('The variance is: ', variance)
    print('The bias is: ', bias)
    print('The expected out-of-sample error is: ', eout)
    print('The variance+bias is: ', variance+bias)

    xs = np.arange(-1, 1, 0.01)
    true_f, avg_gf, var_gf, ubs, lbs = [],[], [], [], []
    for x in xs:
        true_f.append(target_func(x))
        mean_g, var_g, bias_g = avg_g(x, gd_func, num_data_samples, target_func)
        avg_gf.append(mean_g)
        var_gf.append(var_g)
        ubs.append(mean_g + np.sqrt(var_g))
        lbs.append(mean_g - np.sqrt(var_g))
        
    plt.plot(xs, true_f, label='f(x)')
    plt.plot(xs, avg_gf, color='green', label='Problem 2.23: Average Hypothesis g_bar')
    plt.legend(['f(x)', 'g_bar(x)',])
    plt.show()

calculate_bias_var_eout(gd_func,lambda x: x**2,num_data_samples,num_x_samples)
# %%
