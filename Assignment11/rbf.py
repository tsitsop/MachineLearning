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


# return the distance from the potential center to the set of centers
def get_distance_to_centers(potential_center, centers):
    closest_distance = 2.0

    for center in centers:
        distance = get_distance(potential_center[0], center[0])
        if distance < closest_distance:
            closest_distance = distance

    return closest_distance

# return the coordinates of the center that is closest to the point
def get_closest_center(point, centers):
    closest_distance = 2.0

    for center in centers:
        distance = get_distance(point[0], center[0])
        if distance < closest_distance:
            closest_distance = distance
            closest_center = center

    return closest_center


# create the list of centers
def get_centers(data, n, M, centers):
    # centers = list()
    # pick single random poing
    if len(centers) == 0:
        data_point = data[random.randint(0, n-1)]
        centers.append((tuple(data_point[0]), data_point[1]))

    for i in range(len(centers), M):
        distances_to_centers = {}

        # get the distance from each point to the set of centers
        for potential_center in data:
            # if we've already added this point to centers, then skip this point
            if any((potential_center[0] == center[0][0]).all() for center in centers):
                continue

            tuple_potential_center = (tuple(potential_center[0]), potential_center[1])
            distance_to_centers = get_distance_to_centers(potential_center, centers)
            distances_to_centers[tuple_potential_center] = distance_to_centers

        # find furthest point, aka the BEST center
        max_distance = -1
        best_new_center = None
        for potential_center, distance in distances_to_centers.items():
            if distance > max_distance:
                max_distance = distance
                best_new_center = potential_center

        centers.append(best_new_center)

    # center_data, _ = zip(*centers)
    # _, cx1s, cx2s = zip(*center_data)
    # plt.scatter(cx1s, cx2s, s=50, c="red")


    # centers is a list of data points
    return centers


def gaussian_kernel(x):
    return np.e**(-0.5*x**2)


def alpha(x, mu, r):
    term = get_distance(x, mu)/r

    return gaussian_kernel(term)


def create_Z(data, k, r, centers):
    Z = np.zeros((len(data), k+1))

    for i in range(len(data)):
        for j in range(k+1):
            if j == 0:
                Z[i][j] = 1
                continue

            Z[i][j] = alpha(data[i][0], centers[j-1][0], r)

    return Z

# find a better weight through linear regression
def linear_regression(x, y, n):
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    y = y.transpose()

    x_transpose = x.transpose()

    psuedo_inverse_x = np.linalg.inv(x_transpose * x) * x_transpose

    opt = psuedo_inverse_x * y

    opt_array = np.squeeze(np.asarray(opt))

    return opt_array


# determine if point classified correctly
def goodH(w, point, point_class):
    # print(w)
    # print(point)
    result = np.dot(w, point)

    if (np.sign(result) == np.sign(point_class)):
        return True

    return False


def rbf(Z, ys, centers, k, r):
    # fit linear model - train ws etc
    w = linear_regression(Z, ys, len(Z))

    return w


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

    # plot training data
    data, ys = zip(*training_data)
    ys = list(ys)
    _, x1s, x2s = zip(*data)
    
    f, ax = plt.subplots(2,2)

    for i in range(len(data)):
        if ys[i] == 1:
            ax[0][0].plot(x1s[i],x2s[i], 'bo')
        else:
            ax[0][0].plot(x1s[i],x2s[i], 'rx')
            

    centers = list()
    ecvs = list()

    min_error = 1
    min_error_k = 0

    # run RBF
    for k in range(1, 101):
        r = 2/math.sqrt(k)
        
        # calculate centers
        centers = get_centers(training_data, len(training_data), k, centers)
        
        # compute Z
        Z = create_Z(training_data, k, r, centers)

        errors = 0
        # leave one out cv
        for i in range(len(training_data)):
            # remove point i
            current_training_set = list(Z)
            del current_training_set[i]
            current_ys = list(ys)
            del current_ys[i]

            w = rbf(current_training_set, current_ys, centers, k, r)
            
            if goodH(w, Z[i], ys[i]) == False:
                errors += 1

        
        model_error = errors/float(len(training_data))
        ecvs.append(model_error)

        if model_error < min_error:
            min_error = model_error
            min_error_k = k

    # plot ecv vs k
    ks = np.linspace(1, 100, 100)
    ax[0][1].plot(ks, ecvs)
    print("Best number of centers was ", min_error_k, " with cverror of ", min_error)

    # get centers for optimal k
    centers = []
    centers = get_centers(training_data, len(training_data), min_error_k, centers)

    # create optimal Z
    opt_r = 2/math.sqrt(min_error_k)
    opt_Z = create_Z(training_data, min_error_k, opt_r, centers)

    # get optimal weights
    opt_w = rbf(opt_Z, ys, centers, min_error_k, opt_r)

    # in sample error
    error = 0
    for i in range(len(opt_Z)):
        if goodH(opt_w, opt_Z[i], ys[i]) == False:
            errors += 1
    print("It had an ein of ", errors/float(len(training_data)))
    
    

    # plot decision boundaries
    boundary_points = np.random.uniform(-1,1,(10000,2))
    boundary_points = np.column_stack((np.array([1]*10000), boundary_points))
    bp_list = list()

    # give dummy y value
    for point in boundary_points:
        bp_list.append((point, 0))

    transformed_boundary_points = create_Z(bp_list, min_error_k, opt_r, centers)

    for i in range(len(boundary_points)):
        if np.sign(np.dot(opt_w, transformed_boundary_points[i])) == 1:
            ax[1][0].plot(boundary_points[i][1], boundary_points[i][2], 'bo')
        else:
            ax[1][0].plot(boundary_points[i][1], boundary_points[i][2], 'rx')            

    # calculate etest
    test_errors = 0
    transformed_testing_points = create_Z(testing_data, min_error_k, opt_r, centers)
    _, test_ys = zip(*testing_data) 
    test_ys = list(test_ys)
    print(test_ys[0])
    print(transformed_testing_points[0])

    for i in range(len(transformed_testing_points)):
        if goodH(opt_w, transformed_testing_points[i], test_ys[i]) == False:
            test_errors += 1

    test_error = test_errors/float(len(transformed_testing_points))    

    print("It had test error of", test_error)
    
    plt.show()