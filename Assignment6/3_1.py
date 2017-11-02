import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import quad


def generateData(rad, thk, sep, n):
    data = list()

    # Generate all data
    for i in range(n):
        val = random.randint(1,2)
        if (val == 1):
            ###### Create red point ######
            # generate random number from 0 to 180 - represents angle from center
            angle = math.radians(random.uniform(0,180))

            # generate random distance from rad to rad+thk
            distance = random.uniform(rad, rad+thk)

            # generate cartesian coordinates from polar ones
            x1 = distance * math.cos(angle) + (thk + rad)
            x2 = distance * math.sin(angle)

            point = np.array([1, x1, x2])
            data.append((point, -1))

            plt.plot(x1, x2, 'ro')
        else:
            ##### Create blue point #####
            # generate random number from 0 to 180 - represents angle from center
            angle = math.radians(random.uniform(180,360))

            # generate random distance from rad to rad+thk
            distance = random.uniform(rad, rad+thk)

            # generate cartesian coordinates from polar ones
            x1 = distance * math.cos(angle) + (1.5*thk + 2*rad)
            x2 = distance * math.sin(angle) - sep

            point = np.array([1, x1, x2])
            data.append((point, 1))

            plt.plot(x1, x2, 'bo')

    return data



def formula(w, x):
	return ((-w[1]/w[2]) * x - (w[0]/w[2]))

# find optimal weight for linear regression
def linear_regression(data, n):
    x_a = np.zeros(shape=(n,3))
    y_a = np.zeros(shape=(n,1))

    for i in range(n):
        x_a[i] = data[i][0]
        y_a[i] = data[i][1]
        
    x = np.asmatrix(x_a)
    y = np.asmatrix(y_a)

    x_transpose = x.transpose() 

    psuedo_inverse_x = np.linalg.inv(x_transpose * x) * x_transpose

    opt = psuedo_inverse_x * y

    opt_array = np.squeeze(np.asarray(opt))

    return opt_array

# determine if point classified correctly
def goodH(w, point):
    result = np.dot(w, point[0])

    if (np.sign(result) == np.sign(point[1])):
        return True

    return False

#run PLA algorithm
def pla(w, data, n):
    c = 1
    while (True):
        all_correct = True
        
        # pick some misclassified point
        for i in range(n):
            # if point is classified correcly, keep looking
            if (goodH(w, data[i]) == True):
                continue
            
            # if point misclassified, use update rule and stop looking
            w = w + data[i][1] * data[i][0]

            all_correct = False
            break
        
        # if none are misclassified, we found an optimal set of weights
        if (all_correct):
            print c
            return (w, c)

        c += 1
        
        

def main():
    rad = 10
    thk = 5
    sep = 0.1

    # number of points generated
    n = 2000

    # list of number of iterations
    iters = [0] * 25
    
    # for i in range(1,26):
        # weights for PLA
    w = np.array([6, -4, 100])

        # all points in form ([1, x1, x2], y)
    data = generateData(rad, thk, sep, n)
    
        # run the PLA algorithm
    pla_w, p = pla(w, data, n)
        # print pla_w, count

    
    # plt.plot(np.arange(0.2,5.2,0.2), iters)

    # run linear regression algorithm
    # linear_regression_w = linear_regression(data, n)

    # define the xspace we want to show line in
    xspace = np.linspace(0, 3*(rad + thk))

    # plot our optimal weight
    plt.plot(xspace, formula(pla_w, xspace), 'r')
    # plt.plot(xspace, formula(linear_regression_w, xspace), 'b')

    plt.xlabel('sep')
    plt.ylabel('iterations')
    plt.show()






if __name__ == "__main__":
    main()    


