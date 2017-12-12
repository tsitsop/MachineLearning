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


def kernel(x, y):
    return (1+np.dot(np.transpose(x), y))**8

def svm_dual(data, cost):
    xs, ys = zip(*data)
    N = len(data)

    # P is (N)x(N) matrix
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            val = ys[i]*ys[j]*kernel(xs[i], xs[j])
            P.itemset((i,j), val)
    print("P: ", P.shape)

    # q is Nx1 matrix of -1s
    q = -1*np.ones((N, 1))
    print("q: ", q.shape)

    # G is (2N+2)x(N) matrix
    # first two rows satisfy equality constraint
    # final N rows satisfy cost constraint
    G = np.reshape(np.array(ys), (1,N))   # ytranspose
    G = np.vstack((G, -1*G))              # -1*ytranspose
    G = np.vstack((G, -1*np.identity(N))) # -1*NxNidentity
    G = np.vstack((G, np.identity(N)))    # NxNidentity
    print("G: ", G.shape)

    # H is (2N+2)x1 matrix
    # first two rows satisdy equality constraint
    # final N rows satisfy cost constraint
    h = np.zeros((2,1))                     # 2x1 zero vector
    h = np.vstack((h, np.zeros((N,1))))     # Nx1 zero vector
    h = np.vstack((h, cost*np.ones((N,1)))) # Nx1 cost vector
    print("h: ", h.shape)
    print()
    print()


    # convert to cvxopt matrices
    P = P.astype(np.double)
    P = cvxopt.matrix(P)
    q = q.astype(np.double)    
    q = cvxopt.matrix(q)
    G = G.astype(np.double)
    G = cvxopt.matrix(G)
    h = h.astype(np.double)
    h = cvxopt.matrix(h)

    sol = cvxopt.solvers.qp(P, q, G, h)

    print(np.array(sol['x']).shape)

    return sol['x']


def function(w, x, b):
    return np.dot(np.transpose(w),x) + b



if __name__ == '__main__':
    data = list()
    data.append((np.reshape(np.array([1, 0]), (2,1)), 1))
    data.append((np.reshape(np.array([-1, 0]), (2,1)), -1))

    # run svm to get opt weights and bias
    cost = 100
    alphas = svm_dual(data, cost)

    # use alphas to get optimal weights, bias
    xs, ys = zip(*data)
    w = sum([a*y*x for (a,x,y) in zip(alphas,xs,ys)])   

    b = 0.0
    pos_count = 0

    for i in range(len(alphas)):
        if alphas[i] > 0:
            pos_count += 1
            b += ys[i] - np.dot(np.transpose(w), xs[i])

    fig, ax = plt.subplots(2)
    # plot data points
    for point in data:
        if point[1] == 1:
            ax[0].plot(point[0].item(0), point[0].item(1), 'bo')
        else:
            ax[0].plot(point[0].item(0), point[0].item(1), 'rx')
            

    # plot decision boundaries
    boundary_points = np.random.uniform(-1, 1, (1000,2))
    bp_list = list()

    # give dummy y value
    for point in boundary_points:
        bp_list.append((point, 0))
    
    
    
    for i in range(len(bp_list)):
        if np.sign(function(w, boundary_points[i], b)) == 1:
            ax[0].plot(boundary_points[i].item(0), boundary_points[i].item(1), 'bo', alpha=0.14)
        else:
            ax[0].plot(boundary_points[i].item(0), boundary_points[i].item(1), 'rx', alpha=0.14)
        


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









