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
def get_centers(data, n, M):
    centers = list()
    # pick single random poing
    data_point = data[random.randint(0, n-1)]
    centers.append((tuple(data_point[0]), data_point[1]))

    for i in range(1, M):
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

    # centers is a list of data points
    return centers


if __name__ == '__main__':
    n = 1000
    # generate 10,000 uniformly distributed
    #  data points between 0, 1
    data = np.random.rand(n, 2)
    ones = np.ones(n)

    data = np.column_stack((ones,data))


    ys = [1]*n
    zipped = zip(data, ys)
    new_data = list(zipped)
    print(new_data[0])


    data, ys = zip(*new_data)
    _, x1s, x2s = zip(*data)
    plt.scatter(x1s,x2s, s=10, c="black")
    
    centers = get_centers(new_data, n, 10)
    center_data, _ = zip(*centers)
    _, cx1s, cx2s = zip(*center_data)
    plt.scatter(cx1s, cx2s, s=50, c="red")
    
    plt.show()