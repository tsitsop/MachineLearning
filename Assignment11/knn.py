import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
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


def get_distance(point1, point2):
    return np.linalg.norm(point1-point2)


# return the indices of the k nearest neighbors and each of their distances
def k_nearest_neighbor_distances(k, data, new_point):
    distances = list()

    # get the distance to each point
    for data_point in data:
        distances.append(get_distance(new_point, data_point[0]))

    distances = np.array(distances)

    # get indices of k nearest neighbors
    indices = distances.argsort()[:k]

    knn_distances = dict()
    for i in range(indices.size):
        knn_distances[indices[i]] = distances[i]

    # return map where key is index of a nearest neighbor, value is the distance of that neighbor
    return knn_distances


# return the classification based off the k nearest neighbors
def k_nearest_neighbor(k, data, new_point):
    distances = k_nearest_neighbor_distances(k, data, new_point)

    majority = 0
    for k, v in distances.items():
        majority += data[k][1]

    if majority >= 0:
        return 1

    return 0


# gets closest center to data point
def get_closest_center(data, data_index, centers):
    # get distance from point data_index to each center
    distances = []
    for center in centers:
        # if center == -1 then haven't found this center or any after it
        # so we are done finding distances to centers
        if center == -1:
            break
        
        # get the distance from point d to the center
        dist = get_distance(data[data_index], data[center])
        distances.append((dist, center))

    # minimum of distances from point data_index to each center
    #  is the distance from point data_index to the set of centers 
    min_distance = ((2.0, -1), -1)
    for distance in distances:
        if distance[0] < min_distance[0][0]:
            min_distance = (distance, data_index)


    # form:
    #   (distance, data_index)
    #   ((distance from data point 'data_index' to data point 'center', index of data point 'center'), data point 'data_index')
    return min_distance


def get_centers(data, n, M):
    centers = [-1]*M
    centers[0] = random.randint(0, n-1)

    min_distances = list()

    for i in range(1, M):

        # This is the distance from each point to the set of centers
        # It is reset every time we add a new center.
        min_distances = []

        # for each data point
        for d in range(n):
            if d in centers:
                continue
            min_distance = get_closest_center(data, d, centers)

            # (distance from data point d to center i, data point d)
            min_distance = (min_distance[0][0], min_distance[1])

            # add the distance from point d to the set of centers
            #  to our list of distances to set of centers
            min_distances.append(min_distance)

        # at this point, have found distance of each point to the set of centers
        #  the biggest distance is going to be the furthest point from the set

        max_distance = (0, -1)
        for distance in min_distances:
            if distance[0] > max_distance[0]:
                max_distance = distance

        # max_distance[1] is our new center
        centers[i] = max_distance[1]

    # plot M centers
    for c in range(len(centers)):
        if c == 0:
            plt.plot(data[centers[c]][0], data[centers[c]][1], 'bx')
        else:
            plt.plot(data[centers[c]][0], data[centers[c]][1], 'bo')

    return centers


def create_region_data_sets(data, n, centers):
    
    # dictionary where key is center,
    #  value is list of points belonging to it
    region_data = dict()
    for center in centers:
        region_data[center] = list()

    for i in range(n):
        # closest_center will be form ((distance, center), i)
        closest_center = get_closest_center(data, i, centers)
        
        region_data[closest_center[0][1]].append(data[i])

    return region_data        


def get_radii(data, region_data):
    radii = dict()
    for k, v in region_data.items():
        max_distance = 0
        for point in v:
            # get distance between center (data[k])
            #  and point in center's region
            distance = get_distance(data[k], point)

            if distance > max_distance:
                max_distance = distance

        radii[k] = max_distance

    return radii


class Region:
    def __init__(self):
        self.center_index = 0
        self.radius = 0.0
        self.data_points = list()



def create_regions(data, n, M):
    regions = list()

    # list of indices for centers
    centers = get_centers(data, n, M)

    # list of region data sets
    region_data = create_region_data_sets(data, n, centers)

    # list of region radii
    radii = get_radii(data, region_data)

    # create region objects
    for center in centers:
        region = Region()
        region.center_index = center
        region.data_points = region_data[center]
        region.radius = radii[center]
        regions.append(region)

    return regions


def order_regions(data, query_point, regions):
    # regions is a list of region objects

    distances = list()

    for region in regions:
        # distance to region is ((distance from point to region) - region's radius)
        point_to_center = get_distance(query_point, data[region.center_index])
        distance = point_to_center - region.radius

        distances.append((distance, region))

    # order distances by first element in each tuple aka distance
    ordered_distances = sorted(distances, key=lambda element: element[0])

    return ordered_distances 


def branch_and_bound(data, n, k, M, query_points):
    # separate training points from y values
    xs, ys = zip(*data)
    error = 0.0

    # define regions
    #  - includes centers, radii, points
    regions = create_regions(xs, n, M)
    print("defined regions!")

    # need to find kNN for each query_point
    for query_point in query_points:
        nearest_neighbors = []    

        # determine order of regions from closest -> furthest
        #  in form [(distance, region)]
        ordered_regions = order_regions(xs, query_point[0], regions)
        print("ordered region")
        # start at closest region
        current_region_counter = 0

        while len(nearest_neighbors) < k:
            # print("whats going on here...")
            # search region current_region_counter for more nearest neighbors
            #  - num neighbors searching for will depend on the number we've found
            #  nn_possibilities is of form {neighbor_index: distance}
            nn_possibilities = k_nearest_neighbor_distances(k-len(nearest_neighbors), ordered_regions[current_region_counter][1].data_points, query_point[0])
            
            # check each of these (k-j) neighbors to see if they pass bound condition
            for key, v in nn_possibilities.items():
                # if we've reached the final region, nearest neighbor is closest in this region
                if current_region_counter == len(regions)-1:
                    print("reached final region")
                    nearest_neighbors.append((key, v))

                    # remove this point from the region so that it isn't considered in future
                    del ordered_regions[current_region_counter][1].data_points[key]

                    # want to start at closest center when looking for next closest point
                    current_region_counter = 0
                    continue

                # calculate parts of bound equation

                # distance from point to neighbor
                left_bound = v

                # distance from point to next nearest region
                next_region = ordered_regions[current_region_counter+1][1]
                right_bound = get_distance(query_point[0], xs[next_region.center_index]) - next_region.radius 
                
                # on success, add neighbor to nearest_neighbors
                if left_bound <= right_bound:
                    print("yessss")
                    nearest_neighbors.append((key, v))

                    # remove this point from the region so that it isn't considered in future
                    del ordered_regions[current_region_counter][1].data_points[key]

                    # want to start at closest center when looking for next closest point
                    current_region_counter = 0
                # on failure, change to next closest region, break, repeat
                else:
                    current_region_counter += 1
                    break
        
        majority = 0.0
        # see classification based on k nearest neighbors
        for neighbor in nearest_neighbors:
            majority += ys[neighbor[0]]

        # if neighbors classify point incorrectly, have an error
        if np.sign(majority) != np.sign(query_point[1]):
            error += 1

    error = error / len(query_points)
    
    return error



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

    ks = range(1, math.ceil(math.sqrt(300) + 5))
    model_errors = [0.0] * len(ks)


    t = time.time()
    # k models
    for k in ks:
        cv_errors = [0.] * len(training_data)
        print(k)
        # leave one out cross validation
        for i in range(len(training_data)):
            # leave one point out
            data_i = list(training_data)
            del data_i[i]
            # run branch and bound nearest neighbor to find single error
            cv_errors[i] = branch_and_bound(data_i, len(data_i), k, 10, testing_data)
        
        model_errors[k] = sum(cv_errors)/len(model_errors)
    
    print(t-time.time())
    print(model_errors)
    print(min(model_errors))






  # maybe change implementation to stop storing center as an index
  # - could be better to store it, in the end, as an array value