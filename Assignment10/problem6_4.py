import random
import numpy as np
import matplotlib.pyplot as plt
import math

def generateData(rad, thk, sep, n):
    data = list()

    # Generate all data
    for i in range(n):
        val = random.randint(1,2)
        if (val == 1):
            ###### Create red point ######
            # generate random number from 0 to 180 - represents angle from center
            angle = math.radians(random.uniform(0,180))

            # generate random distance from rad to rad+thk
            distance = random.uniform(rad, rad+thk)

            # generate cartesian coordinates from polar ones
            x1 = distance * math.cos(angle) + (thk + rad)
            x2 = distance * math.sin(angle)

            point = np.array([1, x1, x2])
            data.append((point, -1))
        else:
            ##### Create blue point #####
            # generate random number from 0 to 180 - represents angle from center
            angle = math.radians(random.uniform(180,360))

            # generate random distance from rad to rad+thk
            distance = random.uniform(rad, rad+thk)

            # generate cartesian coordinates from polar ones
            x1 = distance * math.cos(angle) + (1.5*thk + 2*rad)
            x2 = distance * math.sin(angle) - sep

            point = np.array([1, x1, x2])
            data.append((point, 1))

    return data


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1-point2)


def k_nearest_neighbor(k, data, new_point):
    ''' Returns the value of new_point based on k nearest neighbor algorithm '''

    distances = list()

    # get the distance to each point
    for point in data:
        distances.append(euclidean_distance(new_point, point[0]))

    distances = np.array(distances)

    # get indices of k nearest neighbors
    indices = distances.argsort()[:k]

    majority = 0
    for i in range(indices.size):
        majority += data[indices.item(i)][1]

    # if majority of k points are positive, this point is positive
    if majority >= 0:
        return 1

    return -1


def plot_nearest_neighbor(k ,data, rad, thk, sep):
    xspace = np.linspace(0,4*rad,50)
    yspace = np.linspace(-(rad+thk+sep),15,50)

    for i in range(len(xspace)):
        for j in range(len(yspace)):
            if k_nearest_neighbor(k, data, [1, xspace[i], yspace[j]]) == 1:
                plt.plot(xspace[i], yspace[j], 'bx', alpha=0.3)
            else:
                plt.plot(xspace[i], yspace[j], 'rx', alpha=0.3)

    for point in data:
        if point[1] == 1:
            plt.plot(point[0][1], point[0][2], 'ro')
        else:
            plt.plot(point[0][1], point[0][2], 'bo')


    plt.xlabel('x1')
    plt.ylabel('x2')

if __name__ == '__main__':
    rad = 10
    thk = 5
    sep = 5

    # number of points generated
    n = 2000

    # all points in form ([1, x1, x2], y)
    data = generateData(rad, thk, sep, n)

    # 1-NN
    plt.title("1-NN")
    plot_nearest_neighbor(1, data, rad, thk, sep)
    plt.show()

    # 3-NN
    plt.title("3-NN")
    plot_nearest_neighbor(3, data, rad, thk, sep)
    plt.show()
