import numpy as np
import cvxopt
from matplotlib import pyplot as plt


# read data from the file
def get_input(filename):
    f = open(filename)
    data = f.read()

    parsedStringData = data.split(" ")

    del parsedStringData[-1]

    parsedData = list(map(float, parsedStringData))

    return parsedData


# clean input so usable structure, 
# return a list in form [number, greyscale values]
def clean_input(data):
    cleanedData = list()

    i = 0
    while i < len(data):
        # array of greyscale vals
        info = [0.]*256
        # each of next 256 values are greyscale so add to info
        for j in range(256):
            info[j] = data[(i+1)+j]

        cleanedData.append([data[i], info])

        i += 257

    return cleanedData


def get_intensities(data):
    # list of intensities for each image
    intensities = [0.]*len(data)

    # find intensity of each image
    for i in range(len(data)):
        intensities[i] = np.sum(data[i][1])

    return intensities


def get_horizontal_symmetries(data):
    # list of symmetries for each number. Symmetry defined by 
    # difference_matrix = absolute value of elementwise_subtraction(left half,flipped right half)
    # sum of difference_matrix = intensity - then find avg, mult by -1
    symmetries = [0.]*len(data)

    for i in range(len(data)):
        # convert to 16d array with 16 elements per row
        imgarray = np.array(data[i][1])
        img_matrix = np.reshape(imgarray, (-1, 16))

        # left intensity is the sum of the pixels in left half of the image.
        left = list()
        for j in range(len(img_matrix)):       # for each line of pixels
            left.append(img_matrix[j][:8])     #   append the left half of the pixels as a row in our list 
        
        # right intensity is the sum of the pixels in right half of the image
        right = list()
        for j in range(len(img_matrix)):       # for each line of pixels
            right.append(img_matrix[j][8:])    #   append right half of pixels

        # reverse the right half to simulate flipped image
        right_flipped = list()
        for j in range(len(right)):
            right_flipped.append(list(reversed(right[j])))            

        # convert to numpy arrays for computation
        left_array = np.array(left)
        right_flipped_array = np.array(right_flipped)

        # difference in each pixel of the two halves
        differences = abs(left_array - right_flipped_array)

        # compute average intensity, mult by -1
        symmetry = -1 * np.sum(differences)

        symmetries[i] = symmetry

    return symmetries


def get_vertical_symmetries(data):
    symmetries = [0.]*len(data)

    for i in range(len(data)):
        # convert to 16d array with 16 elements per row
        imgarray = np.array(data[i][1])
        img_matrix = np.reshape(imgarray, (-1, 16))

        top = img_matrix[:8]
        bottom = img_matrix[8:]

        top_array = np.array(top)
        bottom_array = np.array(bottom)

        ll = list(reversed(bottom_array))
        bottom_flipped_array = np.array(ll)

        differences = abs(top_array - bottom_flipped_array)

        symmetry = -1 * np.sum(differences)

        symmetries[i] = symmetry

    return symmetries


# change raw(ish) input into our traditional data format of [x, y]
def create_data(input):
    data = list()
    
    intensities =  get_intensities(input)
    horizontal_symmetries = np.array(get_horizontal_symmetries(input))
    vertical_symmetries = np.array(get_vertical_symmetries(input))
    
    overall_symmetries = horizontal_symmetries+vertical_symmetries

    # input in form of array[[num, grayscal vals]
    for i in range(len(input)):
        x = np.reshape(np.array([intensities[i], overall_symmetries[i]]), (2, 1))

        if (input[i][0] == 1.0):
            data.append((x, 1))
        else:
            data.append((x, -1))
        
    return data


# data in form [x, y] where x = [1, intensity, symmetry]
def normalize_data(data):
    min_intensity = min(a.item(0) for (a,b) in data)
    max_intensity = max(a.item(0) for (a,b) in data)

    min_symmetry = min(a.item(1) for (a,b) in data)
    max_symmetry = max(a.item(1) for (a,b) in data)


    for i in range(len(data)):
        intensity = data[i][0].item(0)
        data[i][0][0] =\
            2. * (intensity - min_intensity) / (max_intensity - min_intensity) - 1         

        symmetry = data[i][0].item(1)
        data[i][0][1] =\
            2. * (symmetry - min_symmetry) / (max_symmetry - min_symmetry) - 1


def kernel(x, y):
    return (1+np.dot(np.transpose(x), y))**8


def svm(data, cost):
    xs, ys = zip(*data)
    dim = xs[0].size
    N = len(data)

    # Q is (N+d+1)x(N+d+1) matrix
    Q = np.zeros((N+dim+1, N+dim+1))
    for i in range(1, dim+1):
        for j in range(1, dim+1):
            if i == j:
                Q.itemset((i, j), 1)

    # p is (N+d+1)x(1) vector
    p = np.zeros((N+dim+1, 1))
    for i in range(dim, p.size):
        p.itemset(i, cost)

    # A is (d+1)x(2N) matrix
    A = np.squeeze(np.array([y*np.transpose(x) for (y, x) in data]))
    A = np.hstack((np.reshape(np.array(ys), (N, 1)), A))
    A = np.hstack((A, np.identity(N)))

    temp = np.hstack((np.zeros((N, dim+1)), np.identity(N)))
    A = np.vstack((A, temp))

    # c is a (2N)x(1) vector
    c = np.ones((N, 1))
    c = np.vstack((c, np.zeros((N, 1))))

    print("Q ", Q.shape)
    print("p ", p.shape)
    print("A ", A.shape)
    print("c ", c.shape)

    Q = Q.astype(np.double)
    Q = cvxopt.matrix(Q)
    p = p.astype(np.double)
    p = cvxopt.matrix(p)
    A = A.astype(np.double)
    A = cvxopt.matrix(-A)
    c = c.astype(np.double)
    c = cvxopt.matrix(-c)

    sol = cvxopt.solvers.qp(Q, p, A, c)

    print(np.array(sol['x']).shape)

    return sol['x']


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
    h = np.zeros((1,1))                     # 1x1 zero vector
    h = np.vstack((h, -1*np.zeros((1,1))))  # 1x1 -1*zero vector
    h = np.vstack((h, -1*np.zeros((N,1))))  # Nx1 -1*zero vector
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

    print(sol)

    return sol['x']



def function(w, x, b):
    return np.dot(np.transpose(w),x) + b



if __name__ == '__main__':
    # get all data from files
    train = get_input('ZipDigits.train')
    test = get_input('ZipDigits.test')
    combined = train+test

    # clean the input to organize structure
    cleaned_input = clean_input(combined)

    # get data to be in terms of intensity and symmetries
    data = create_data(cleaned_input)

    # normalize ALL data
    normalize_data(data)

    training_data = data[:300]
    testing_data = data[300:]

    # run svm to get opt weights and bias
    cost = 0.001
    
    # output = svm(training_data, cost)

    # b = output[:1]
    # w = output[1:3]
    # xis = output[3:]

    # # plot stuff    
    # fig, ax = plt.subplots(2)
    # for point in training_data:
    #     if point[1] == 1:
    #         ax[0].plot(point[0].item(0), point[0].item(1), 'bo')
    #         # ax[1].plot(point[0].item(0), point[0].item(1), 'bo')            
    #     else:
    #         ax[0].plot(point[0].item(0), point[0].item(1), 'rx')
    #         # ax[1].plot(point[0].item(0), point[0].item(1), 'rx')
            
    # # plot decision boundaries
    # boundary_points = np.random.uniform(-1, 1, (1000,2))
    # bp_list = list()

    # # give dummy y value
    # for point in boundary_points:
    #     bp_list.append((point, 0))
    
        
    # for i in range(len(bp_list)):
    #     if np.sign(function(w, boundary_points[i], b)) == 1:
    #         ax[0].plot(boundary_points[i].item(0), boundary_points[i].item(1), 'bo', alpha=0.14)
    #     else:
    #         ax[0].plot(boundary_points[i].item(0), boundary_points[i].item(1), 'rx', alpha=0.14)
        

    # plt.show()
    
    
    
    
    
    alphas = svm_dual(training_data, cost)

    # use alphas to get optimal weights, bias
    xs, ys = zip(*training_data)
    w = sum([a*y*x for (a,x,y) in zip(alphas,xs,ys)])   

    b = 0.0
    pos_count = 0

    for i in range(len(alphas)):
        if alphas[i] > 0:
            pos_count += 1
            b += ys[i] - np.dot(np.transpose(w), xs[i])
    
    print("pos_count = ", pos_count)
    # plot stuff    
    fig, ax = plt.subplots(2)
    for point in training_data:
        if point[1] == 1:
            ax[0].plot(point[0].item(0), point[0].item(1), 'bo')
            # ax[1].plot(point[0].item(0), point[0].item(1), 'bo')            
        else:
            ax[0].plot(point[0].item(0), point[0].item(1), 'rx')
            # ax[1].plot(point[0].item(0), point[0].item(1), 'rx')
            

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




