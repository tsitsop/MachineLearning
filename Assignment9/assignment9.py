import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import quad

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
        x = np.array([1, intensities[i], overall_symmetries[i]])

        if (input[i][0] == 1.0):
            data.append((x, 1))
        else:
            data.append((x, -1))
        
    return data


# data in form [x, y] where x = [1, intensity, symmetry]
def normalize_data(data):
    min_intensity = min(a[1] for (a,b) in data)
    max_intensity = max(a[1] for (a,b) in data)

    min_symmetry = min(a[2] for (a,b) in data)
    max_symmetry = max(a[2] for (a,b) in data)


    for i in range(len(data)):
        intensity = data[i][0][1]
        data[i][0][1] =\
            2. * (intensity - min_intensity) / (max_intensity - min_intensity) - 1         

        symmetry = data[i][0][2]
        data[i][0][2] =\
            2. * (symmetry - min_symmetry) / (max_symmetry - min_symmetry) - 1

        # if (data[i][1] == 1.):
        #     plt.plot(data[i][0][1], data[i][0][2], 'bo', mew=2, ms=10, fillstyle='none', markeredgecolor='blue')
        # else:
        #     plt.plot(data[i][0][1], data[i][0][2], 'rx', mew=2, ms=10)


# computes legendre transform of x^k
def lt(x, k):
    # base cases
    if (k == 0):
        return 1
    elif (k == 1):
        return x

    # recursive step
    first_term = ((2.0*k - 1.0) / k)*x * lt(x, k-1)
    second_term = ((k - 1.0)/k) * lt(x, k-2)
    
    return first_term - second_term 


# linear regression with decay. l is lambda
def linear_regression(data, n, l):
    x_a = np.zeros(shape=(n,len(data[0][0])))
    y_a = np.zeros(shape=(n,1))

    for i in range(n):
        x_a[i] = data[i][0]
        y_a[i] = data[i][1]
        
    z = np.asmatrix(x_a)
    y = np.asmatrix(y_a)

    z_transpose = z.transpose()

    li = l * np.identity(len(data[0][0]))
    
    inv = np.linalg.inv((z_transpose * z) + li)

    opt = inv * z_transpose * y

    opt_array = np.squeeze(np.asarray(opt))

    return opt_array


# polynomial feature transform
def polynomial_transform(point, degree):
    transformed_point = list()
    x1 = point[1]
    x2 = point[2]

    for i in range(degree+1):
        for j in range(i+1):
            transformed_point.append(x1**(i-j)*x2**j)

    return transformed_point


# legendre feature transform
def legendre_transform(point, degree):
    transformed_point = list()
    x1 = point[1]
    x2 = point[2]

    for i in range(degree+1):
        for j in range(i+1):
            transformed_point.append(lt(x1, i-j)*lt(x2,j))

    return transformed_point


def get_cv_error(data, l, g):
    n = len(data)
    
    x_a = np.zeros(shape=(n,len(data[0][0])))
    y_a = np.zeros(shape=(n,1))
    g = np.array(g)

    for i in range(n):
        x_a[i] = data[i][0]
        y_a[i] = data[i][1]
        
    z = np.asmatrix(x_a)
    y = np.asmatrix(y_a)

    z_transpose = z.transpose()

    li = l * np.identity(len(g))
    
    inv = np.linalg.inv((z_transpose * z) + li)
    
    H = z * inv * z_transpose

    y_hat = H * y

    total_error = 0.0
    for i in range(n):
        diff = y_hat[i] - y[i]
        
        err = (diff / (1.0 - H.item(i,i)))
        
        squared_err = err.item(0)**2
        
        total_error += squared_err

    return total_error/float(n)


def get_test_error(test_data, w):
    total_error = 0.0
    n = len(test_data)

    x_a = np.zeros(shape=(n,len(test_data[0][0])))
    y_a = np.zeros(shape=(n,1))

    for i in range(n):
        x_a[i] = test_data[i][0]
        y_a[i] = test_data[i][1]
        
    x = np.asmatrix(x_a)
    y = np.asmatrix(y_a)
    w = np.asmatrix(w).transpose()

    y_hat = x*w

    for i in range(n):
        err = (y_hat.item(i) - y.item(i))**2
        total_error += err
    
    return total_error/n


# determine if point classified correctly
def goodH(w, point):
    result = np.dot(w, point[0])

    if (np.sign(result) == np.sign(point[1])):
        return True

    return False


# get the number of misclassified points
def get_misclassified(w, data, n):
    err_count = 0

    # for each point in the data set, check if point misclassified
    for i in range(n):
        # if point misclassified, increment error count
        if (goodH(w, data[i]) == False):
            err_count += 1

    return err_count


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
    
    # split into training and test sets
    random.shuffle(data)
    training_data = data[:300]
    testing_data = data[300:]

    # transform ALL data to degree order legendre transform
    degree = 8
    transformed_data = list()
    transformed_test_data = list()
    for i in range(len(training_data)):
        transformed_data.append((np.array(legendre_transform(training_data[i][0], degree)), training_data[i][1]))
        
    for i in range(len(testing_data)):
        transformed_test_data.append((np.array(legendre_transform(testing_data[i][0], degree)), testing_data[i][1]))


    ############### Compute Regression Weights ###############

    # we have 200 different lambdas to try so 200 models
    models = [None]*201
    cv_errors = [0.]*201
    test_errors = [0.]*201
    min_index = 0

    for i in range(201):
        l = i/100.
        g = linear_regression(transformed_data, len(transformed_data), l)
        models[i] = (g, l)
        cv_errors[i] = get_cv_error(transformed_data, l, g)
        test_errors[i] = get_test_error(transformed_test_data, g)

        if cv_errors[i] < cv_errors[min_index]:
            min_index = i

    # get index of smallest cv_error (best model)
    # min_index = np.argmin(cv_errors)
    opt_model = models[min_index]

    # calculate classification error
    e_test = float(get_misclassified(opt_model[0], transformed_test_data, len(transformed_test_data)))  / len(transformed_test_data)
    error_bar_test = math.sqrt(1./(2*len(transformed_test_data)) * np.log(2./0.05))
    e_out = e_test + error_bar_test

    ############### Plot Data ###############

    # plot stuff
    x1 = np.linspace(-1.5, 1.5, 100)
    x2 = np.linspace(-1.5, 1.5, 100)
    x1, x2 = np.meshgrid(x1, x2)

    # plot line weights when lambda = 0
    for i in range(len(training_data)):
        if (training_data[i][1] == 1.):
            plt.plot(training_data[i][0][1], training_data[i][0][2], 'bo', mew=2, ms=10, fillstyle='none', markeredgecolor='blue')
        else:
            plt.plot(training_data[i][0][1], training_data[i][0][2], 'rx', mew=2, ms=10)

    plt.contour(x1, x2, sum([a*b for a,b in zip(legendre_transform([1,x1,x2], degree), models[0][0])]), [0])
    plt.title('Lambda = 0')
    plt.xlabel("intensity")
    plt.ylabel("symmetry")
    plt.show()

    # plot line weights when lambda = 2
    for i in range(len(training_data)):
        if (training_data[i][1] == 1.):
            plt.plot(training_data[i][0][1], training_data[i][0][2], 'bo', mew=2, ms=10, fillstyle='none', markeredgecolor='blue')
        else:
            plt.plot(training_data[i][0][1], training_data[i][0][2], 'rx', mew=2, ms=10)

    plt.contour(x1, x2, sum([a*b for a,b in zip(legendre_transform([1,x1,x2], degree), models[-1][0])]), [0])
    plt.title('Lambda = 2')
    plt.xlabel("intensity")
    plt.ylabel("symmetry")
    plt.show()



    # plot optimal weights
    for i in range(len(training_data)):
        if (training_data[i][1] == 1.):
            plt.plot(training_data[i][0][1], training_data[i][0][2], 'bo', mew=2, ms=10, fillstyle='none', markeredgecolor='blue')
        else:
            plt.plot(training_data[i][0][1], training_data[i][0][2], 'rx', mew=2, ms=10)

    
    opt = min_index/100
    
    plt.title('Optimal lambda = %f' % (opt))
    plt.xlabel("intensity")
    plt.ylabel("symmetry")
    plt.contour(x1, x2, sum([a*b for a,b in zip(legendre_transform([1,x1,x2], degree), opt_model[0])]), [0])
    plt.show()

    # plot errors vs lambda
    x_space = np.linspace(0,2,201)
    plt.plot(x_space, cv_errors, label="cv_error")
    plt.plot(x_space, test_errors, label="test_error")
    plt.xlabel('lambda')
    plt.ylabel('error')
    plt.legend()
    plt.show()
    
    print("E_out <=", e_out)