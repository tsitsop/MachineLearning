import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import quad

def function(x,y):
    return x**2 + 2*y**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def get_gradient(x,y):
    g_x = 2*(2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y) + x)
    g_y = 4*(np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) + y)

    return np.array([g_x, g_y])

def gradient_descent(pt, eta, num_iters):
    w = np.ndarray(shape=(num_iters, 2), dtype=float)
    
    w[0] = pt
    min_point = w[0]
    min_val = function(w[0][0], w[0][1])

    for i in range(num_iters-1):
        g = get_gradient(w[i][0], w[i][1])
        v = -1 * g
        w[i+1] = w[i] + eta*v

        next_val = function(w[i+1][0], w[i+1][1])
        if (next_val < min_val):
            min_val = next_val
            min_point = w[i+1]

    return w, min_val, min_point

def main():
    num_iters = 50
    pt = np.array([0.1, 0.1])
    eta = 0.01
    w,_,_  = gradient_descent(pt, eta, num_iters)

    xspace = np.linspace(0,num_iters, num_iters)
    functions = [function(weight[0], weight[1]) for weight in w]
    
    plt.plot(xspace, functions)
    plt.xlabel('iterations')
    plt.ylabel('f(x, y)')
    plt.show()

    min_vals = np.array([0.]*4)
    min_points = np.ndarray(shape=(4,2), dtype=float)

    pt = np.array([0.1, 0.1])
    _,min_vals[0], min_points[0] = gradient_descent(pt, eta, num_iters)

    pt = np.array([1., 1.])
    _,min_vals[1], min_points[1] = gradient_descent(pt, eta, num_iters)

    pt = np.array([-0.5, -0.5])
    _,min_vals[2], min_points[2] = gradient_descent(pt, eta, num_iters)

    pt = np.array([-1., -1.])
    _,min_vals[0], min_points[0] = gradient_descent(pt, eta, num_iters)

    print min_vals
    print min_points

if __name__ == '__main__':
    main()
