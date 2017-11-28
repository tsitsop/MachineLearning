import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math
from scipy.integrate import quad
import operator

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


# get euclidean distance from point1 to point2
def get_distance(point1, point2):
    return np.linalg.norm(point1-point2)


# return the classification based off the k nearest neighbors
def k_nearest_neighbor(k, data, test_point_index, distance_matrix):
    # each row of distance matrix is the distances of all other points to this point
    distances = distance_matrix[test_point_index]

    # get indices of k nearest neighbors
    indices = distances.argsort()[1:k+1]

    majority = 0
    for index in indices:
        if index == test_point_index:
            continue
        if data[index][1] == 1:
            majority += 1
        else:
            majority -= 1

    sign = np.sign(majority)

    return sign


def create_distance_matrix(data1, data2):
    distances = np.zeros((len(data1), len(data2)))
    for i in range(len(data1)):
        for j in range(len(data2)):
            distances[i][j] = get_distance(data1[i][0], data2[j][0])

    return distances

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


    ############### Run KNN CV ###############

    ks = range(1, 100)
    model_errors = [0.0] * len(ks)

    training_distance_matrix = create_distance_matrix(training_data, training_data)

    cv_errors = [0.]*len(ks)
    # k models
    for k in ks:
        # leave one out cross validation
        for left_out_index in range(len(training_data)):
            # run knn to find if classify left out point correctly
            classification = k_nearest_neighbor(k, training_data, left_out_index, training_distance_matrix)
            
            if classification != training_data[left_out_index][1]:
                cv_errors[k-1] += 1

        cv_errors[k-1] /= len(training_data)

    opt_k = cv_errors.index(min(cv_errors))


    f, ax = plt.subplots(2,2)

    # plot training data
    xs, ys = zip(*training_data)
    _,x1s,x2s = zip(*xs)

    for i in range(len(training_data)):
        if ys[i] == 1:
            ax[0][0].plot(x1s[i],x2s[i], 'bo')
        else:
            ax[0][0].plot(x1s[i],x2s[i], 'rx')

    # plot ecv vs k
    ax[0][1].plot(ks, cv_errors)
    print("It had an optimal k of", opt_k, " with an ecv of ", cv_errors[opt_k])

    # in sample error
    # currently ein same as ecv optimal
    print("It had an ein of ", cv_errors[opt_k])
    
    # plot decision boundaries
    boundary_points = np.random.uniform(-1,1,(5000,2))
    boundary_points = np.column_stack((np.array([1]*5000), boundary_points))
    bp_list = list()

    # give dummy y value
    for point in boundary_points:
        bp_list.append((point, 0))

    boundary_distance_matrix = create_distance_matrix(bp_list, training_data)

    # plot boundary
    for i in range(len(boundary_points)):
        if k_nearest_neighbor(opt_k, training_data, i, boundary_distance_matrix) == 1:
            ax[1][0].plot(boundary_points[i][1], boundary_points[i][2], 'bo')
        else:
            ax[1][0].plot(boundary_points[i][1], boundary_points[i][2], 'rx')            


    # calculate etest
    test_errors = 0
    _, test_ys = zip(*testing_data) 
    test_ys = list(test_ys)
    
    testing_distance_matrix = create_distance_matrix(testing_data, training_data)

    for i in range(len(testing_data)):
        if k_nearest_neighbor(opt_k, training_data, i, testing_distance_matrix) != test_ys[i]:
            test_errors += 1

    test_error = test_errors/float(len(testing_data))    

    print("It had test error of", test_error)
    
    plt.show()





  # maybe change implementation to stop storing center as an index
  # - could be better to store it, in the end, as an array value