import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import quad

# formula used to map w to xspace
def formula(w, x):
	return ((-w[1]/w[2]) * x - (w[0]/w[2]))


# read data from the file
def get_input(filename):
    f = open(filename)
    data = f.read()

    parsedStringData = data.split(" ")

    del parsedStringData[-1]

    parsedData = map(float, parsedStringData)

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

        if (data[i][1] == 1.):
            plt.plot(data[i][0][1], data[i][0][2], 'bo', mew=3, ms=10, fillstyle='none', markeredgecolor='blue')
        else:
            plt.plot(data[i][0][1], data[i][0][2], 'rx', mew=3, ms=10)


# computes legendre transform of (x)^k
def legendre_transform(x, k):
    # base cases
    if (k == 0):
        return 1
    elif (k == 1):
        return x

    # recursive step
    first_term = ((2.0*k - 1.0) / k) * legendre_transform(x, k-1)
    second_term = ((k - 1.0)/k) * legendre_transform(x, k-2)
    
    return first_term - second_term 

# linear regression with decay. l is lambda
def linear_regression(data, n, l):
    x_a = np.zeros(shape=(n,3))
    y_a = np.zeros(shape=(n,1))

    for i in range(n):
        x_a[i] = data[i][0]
        y_a[i] = data[i][1]
        
    x = np.asmatrix(x_a)
    y = np.asmatrix(y_a)

    x_transpose = x.transpose() 

    # modified to add lambda
    psuedo_inverse_x = np.linalg.inv(x_transpose * x + l*np.identity(3)) * x_transpose

    opt = psuedo_inverse_x * y

    opt_array = np.squeeze(np.asarray(opt))

    return opt_array

def polynomial_transform(point, degree_transform):
    new_point = [1]
    x1 = point[1]
    x2 = point[2]

    # keeps track of the previous degree transform's values added
    prev_degree = new_point
    # keeps track of the current degree transform's values added
    cur_degree = list()

    # need to do this for each degree transform
    for i in range(1, degree_transform+1):
        # need to iterate over all the values added in the previous degree
        for j in range(len(prev_degree)):
            # append all updated values from previous degree 
            cur_degree.append(prev_degree[j] * x1)
        # need to also add updated final value with x2
        
        cur_degree.append(prev_degree[-1]*x2)
        # add all values we just created in cur_degree to new point
        new_point += cur_degree

        prev_degree = cur_degree[:]
        cur_degree[:] = []

    print new_point


if __name__ == '__main__':
    # # indicates which polynomial transform using
    # degree_transform = 1

    #  # get all data from files
    # train = get_input('ZipDigits.train')
    # test = get_input('ZipDigits.test')
    # combined = train+test
    
    # # clean the input to organize structure
    # cleaned_input = clean_input(combined)

    # # get data to be in terms of intensity and symmetries
    # data = create_data(cleaned_input)

    # # normalize data
    # normalize_data(data)
    
    # # split into training and test sets
    # random.shuffle(data)
    # training_data = data[:300]
    # testing_data = data[300:]

    # # don't need Legendre right now because in 2d


    # # calculate decayed linear regression weights, where l = lambda
    # l = 0
    # w = linear_regression(training_data, len(training_data), l)

    
    
    # plot our optimal weight
    # xspace = np.linspace(-1, 1, 100)
    # plt.plot(xspace, formula(w, xspace), color='black', label='Training Separator')
    # plt.show()