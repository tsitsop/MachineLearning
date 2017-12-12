import numpy as np
import cvxopt
from matplotlib import pyplot as plt
import operator
import time


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


def svm_dual(data, cost):
    xs, ys = zip(*data)
    N = len(data)

    # P is (N)x(N) matrix
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            val = ys[i]*ys[j]*kernel(xs[i], xs[j])
            P.itemset((i,j), val)

    # q is Nx1 matrix of -1s
    q = -1*np.ones((N, 1))

    # G is (2N)x(N) matrix
    # First N rows satisyfy positivity constraint
    # final N rows satisfy cost constraint
    G = -1*np.identity(N)               # -1*NxNidentity
    G = np.vstack((G, np.identity(N)))  # NxNidentity

    # H is (2N+2)x1 matrix
    # first N rows satisfy positivity constraint
    # final N rows satisfy cost constraint
    h = -1*np.zeros((N,1))  # Nx1 -1*zero vector
    h = np.vstack((h, cost*np.ones((N,1)))) # Nx1 cost vector
    
    # A is (1xN) matrix of ys
    A = np.reshape(ys, (1,N))
    # b is 1x1 0 matrix
    b = np.zeros((1,1))

    # convert to cvxopt matrices
    P = P.astype(np.double)
    P = cvxopt.matrix(P)
    q = q.astype(np.double)    
    q = cvxopt.matrix(q)
    G = G.astype(np.double)
    G = cvxopt.matrix(G)
    h = h.astype(np.double)
    h = cvxopt.matrix(h)
    A = A.astype(np.double)
    A = cvxopt.matrix(A)
    b = b.astype(np.double)
    b = cvxopt.matrix(b)

    sol = cvxopt.solvers.qp(P, q, G, h, A, b)

    return sol['x']


def function(w, x, b):
    return np.dot(np.transpose(w),x) + b


def get_classification(point, alphas, xs, ys, b):
    sum = 0.0
    for i in range(len(alphas)):
        if alphas[i] > 0:
            sum += alphas[i]*ys[i]*kernel(xs[i], point)

    return np.sign(sum+b)


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

    xs, ys = zip(*training_data)

    cv_errors = list()

    t = time.time()
    # try costs between 0.001
    cvxopt.solvers.options['show_progress'] = False
    for cost in range(1, 752, 50):
        error = 0
        for left_out_index in range(len(training_data)):
            current_data = list(training_data)
            del current_data[left_out_index]
            
            alphas = svm_dual(current_data, cost/1000.0)

            right = 0.0
            s_index = 0
            for i in range(len(alphas)):
                if alphas[i] > 0:
                    if alphas[i] < cost:
                        s_index = i
                        break

            for i in range(len(alphas)):
                if alphas[i] > 0:
                    right += ys[i]*alphas[i]*kernel(xs[i], xs[s_index])

            b = ys[s_index] - right

            # if we misclassified the point, add error
            if get_classification(training_data[left_out_index][0], alphas, xs, ys, b) != training_data[1]:
                error += 1

        cv_errors.append(error/float(len(training_data)))
    print(t-time.time())

    # get best error
    best_index = min(enumerate(cv_errors), key=operator.itemgetter(1))[0] 
    opt_cost = (1+best_index*50)/1000.0
    alphas = svm_dual(training_data, opt_cost)
    right = 0.0
    s_index = 0
    for i in range(len(alphas)):
        if alphas[i] > 0:
            if alphas[i] < opt_cost:
                s_index = i
                break

    for i in range(len(alphas)):
        if alphas[i] > 0:
            right += ys[i]*alphas[i]*kernel(xs[i], xs[s_index])

    b = ys[s_index] - right

    # plot training data
    for point in training_data:
        if point[1] == 1:
            plt.plot(point[0][0], point[0][1], 'bo')
        else:
            plt.plot(point[0][0], point[0][1], 'rx')


    # plot decision boundaries
    boundary_points = list()
    for i in range(-100, 100, 2):
        for j in range(-100, 100, 2):
            boundary_points.append(np.reshape(np.array([i/100.0, j/100.0]), (2,1)))

    for i in range(len(boundary_points)):
        if get_classification(boundary_points[i], alphas, xs, ys, b) == 1:
            plt.plot(boundary_points[i].item(0), boundary_points[i].item(1), 'bo', alpha=0.14)
        else:
            plt.plot(boundary_points[i].item(0), boundary_points[i].item(1), 'rx', alpha=0.14)

    test_error = 0
    for point in testing_data:
        if get_classification(point[0], alphas, xs, ys, b) != point[1]:
            test_error += 1
    print("optimal cost: ", opt_cost)
    print("Test error: ", test_error/float(len(testing_data)))

    plt.show()



'''
    linear 8th order:


    knn


    rbf


    nn


    svm


    IF use all methods well can get comparable performances. Can't simply boost performance by changing your method out of these

'''