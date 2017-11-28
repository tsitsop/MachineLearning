import random
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import operator

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



def create_regions(data, n, k, M):
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
    # define regions
    #  - includes centers, radii, points
    regions = create_regions(data, n, k, M)

    # need to find kNN for each query_point
    for query_point in query_points:
        nearest_neighbors = []    

        # determine order of regions from closest -> furthest
        #  in form [(distance, region)]
        ordered_regions = order_regions(data, query_point, regions)

        # start at closest region
        current_region_counter = 0

        while len(nearest_neighbors) < k:
            # search region current_region_counter for more nearest neighbors
            #  - num neighbors searching for will depend on the number we've found
            #  nn_possibilities is of form {neighbor_index: distance}
            nn_possibilities = k_nearest_neighbor_distances(k-len(nearest_neighbors), ordered_regions[current_region_counter][1].data_points, query_point)

            # check each of these (k-j) neighbors to see if they pass bound condition
            for key, v in nn_possibilities.items():
                # calculate parts of bound equation
                left_bound = get_distance(query_point, v)

                next_region = ordered_regions[current_region_counter+1][1]
                right_bound = get_distance(query_point, data[next_region.center_index]) - next_region.radius 
                
                # on success, add neighbor to nearest_neighbors, look at next possibility in this 
                if left_bound <= right_bound:
                    nearest_neighbors.append((key, v))
                # on failure, change to next closest region, break, repeat
                else:
                    current_region_counter += 1
                    break


# run k nearest neighbor on all query points
def brute_force(data, n, k, query_points):
    start_time = time.time()

    #find nearest neighbor for every query point
    for query_point in query_points:
        classification = k_nearest_neighbor(1, data, query_point)

    stop_time = time.time()

    return stop_time - start_time


if __name__ == '__main__':
    n = 300
    # generate 10,000 uniformly distributed
    #  data points between 0, 1
    data = np.random.rand(n, 2)

    # data = np.ndarray((n,2))

    # c = 0
    # for i in range(100):
    #     for j in range(100):
    #         data[c] = np.array([i/100., j/100.])
    #         c += 1

    # for point in data:
    #     plt.plot(point[0], point[1], 'rx')


    # create 10,000 query points
    query_points = np.random.rand(8900, 2)

    # run brute force algorithm and return time taken
    brute_time = brute_force(data, n, 1, query_points)

    # run branch and bound algorithm and return time taken
    # t = time.time()
    # branch_and_bound(data, n, 1, 10, query_points)
    # t2 = time.time()
    # print(t2-t)
    print(brute_time)


    plt.show()