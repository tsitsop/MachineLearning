import numpy as np
import cvxopt
from matplotlib import pyplot as plt

def z_transform(data):
    z = list()

    for point in data:
        z1 = point[0][0]**3 - point[0][1]
        z2 = point[0][0]*point[0][1]
        z.append((np.array([z1, z2]), point[1]))

    return z


def svm(data, dim):
    # p = (d+1)x(1)
    p = np.zeros((dim+1, 1))
    c = np.ones((len(data), 1))

    # mapping program's to mine
    #  P=Q, q=p, G=-A, h=-c

    # Q is (d+1)x(d+1) matrix with diagonal of 1s except top left
    Q = np.zeros((p.size, p.size))
    for i in range(1, p.size):
        for j in range(1, p.size):
            if i == j:
                Q.itemset((i, j), 1)
    
    # A is (N)x(d+1) matrix
    A = list()
    for i in range(len(data)):
        row = np.hstack((np.reshape(np.array([data[i][1]]), (1,1)), data[i][1]*np.transpose(data[i][0])))
        A.append(row)

    A = np.array(A)
    A = np.squeeze(A)

    # convert to cvxopt matrices
    Q = Q.astype(np.double)
    Q = cvxopt.matrix(Q)
    p = p.astype(np.double)    
    p = cvxopt.matrix(p)
    A = A.astype(np.double)
    A = cvxopt.matrix(-A)
    c = c.astype(np.double)
    c = cvxopt.matrix(-c)


    sol = cvxopt.solvers.qp(Q, p, A, c)

    print(sol['x'])
    return sol['x'][0], np.array(sol['x'][1:])

def function(w, x, b):
    return np.dot(np.transpose(w),x) + b

def kernel(x, y):
    return x[0]**3*y[0]**3 - x[0]**3*y[1] - y[0]**3*x[1] + x[1]*y[1] + x[0]*x[1]*y[0]*y[1]


if __name__ == '__main__':
    data = list()
    data.append((np.reshape(np.array([1, 0]), (2,1)), 1))
    data.append((np.reshape(np.array([-1, 0]), (2,1)), -1))

    # run svm to get opt weights and bias
    dim = 2
    b, w = svm(data, dim)

    # run svm with transformed data
    tb, tw = svm(z_transform(data), dim)

    fig, ax = plt.subplots(2)
    # plot data points
    for point in data:
        if point[1] == 1:
            ax[0].plot(point[0].item(0), point[0].item(1), 'bo')
            ax[1].plot(point[0].item(0), point[0].item(1), 'bo')            
        else:
            ax[0].plot(point[0].item(0), point[0].item(1), 'rx')
            ax[1].plot(point[0].item(0), point[0].item(1), 'rx')
            

    # plot decision boundaries
    boundary_points = list()
    for i in range(-100, 100, 2):
        for j in range(-100, 100, 2):
            boundary_points.append((np.reshape(np.array([i/100.0, j/100.0]), (2,1)), 0))
    
    z_boundary_points = z_transform(boundary_points)
    
    for i in range(len(boundary_points)):
        if np.sign(function(w, boundary_points[i][0], b)) == 1:
            ax[0].plot(boundary_points[i][0].item(0), boundary_points[i][0].item(1), 'bx', alpha=0.08)
        else:
            ax[0].plot(boundary_points[i][0].item(0), boundary_points[i][0].item(1), 'rx', alpha=0.08)
        
        if np.sign(function(tw, z_boundary_points[i][0], tb)) == 1:
            ax[1].plot(boundary_points[i][0].item(0), boundary_points[i][0].item(1), 'bx', alpha=0.08)
        else:
            ax[1].plot(boundary_points[i][0].item(0), boundary_points[i][0].item(1), 'rx', alpha=0.08)



    plt.show()



    # a. equation of hyperplane = undefined - the bias is about 0 and weights are [1,0]^T
    # b. 
    #   i. data points unchanged
    #   ii. same as before
    # c. need to save with good stuff
    # d. K(x,y) = [z11 z12] [z21] = z11*z21 + z12*z22
    #                       [z22]
    #           = (x11^3-x12)(x21^3-x22) + (x11x12)(x21x22)
    #           = x11^3*x21^3 - x11^3*x22 - x21^3*x12 + x12*x22 +x11*x12*x21*x22 




