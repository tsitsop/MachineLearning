import numpy as np
import matplotlib.pyplot as plt
import math


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1-point2)


def k_nearest_neighbor(k, data, new_point):
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


def transform_point(point):
    x1 = point[0]
    x2 = point[1]

    z1 = math.sqrt(x1**2 + x2**2)
    z2 = math.atan(x2/(x1+10**(-6)))

    return np.array([z1, z2])


def transform_data(data):
    transformed_data = list()

    for point in data:
        transformed_point = transform_point(point[0])
        transformed_data.append((transformed_point, point[1]))

    return transformed_data


def plot_nearest_neighbor(k , original_data, transform):
    xspace = np.linspace(-5,5,50)
    yspace = np.linspace(-5,5,50)

    # transform data if need be
    data = original_data
    if bool(transform):
        data = transform_data(data)

    for i in range(len(xspace)):
        for j in range(len(yspace)):
            # if we are using the transform, need to transform the
            # test point we are sending into nearest neighbor
            if bool(transform):
                new_point = transform_point([xspace[i], yspace[j]])
            else:
                new_point = np.array([xspace[i], yspace[j]])

            # run k nearest neighbors
            #  passing in transforned data and transformed point if true
            if k_nearest_neighbor(k, data, new_point) == 1:
                plt.plot(xspace[i], yspace[j], 'bx', alpha=0.3)
            else:
                plt.plot(xspace[i], yspace[j], 'rx', alpha=0.3)

    for point in original_data:
        if point[1] == 1:
            plt.plot(point[0][0], point[0][1], 'bo')
        else:
            plt.plot(point[0][0], point[0][1], 'rx')

    plt.xlabel('x1')
    plt.ylabel('x2')


if __name__ == '__main__':
    data = np.array([
        (np.array([1., 0.]), -1),
        (np.array([0., 1.]), -1),
        (np.array([0., -1.]), -1),
        (np.array([-1., 0.]), -1),
        (np.array([0., 2.]), 1),
        (np.array([0., -2.]), 1),
        (np.array([-2., 0.]), 1)
    ])

    # 1-NN decision regions
    plot_nearest_neighbor(1, data, False)

    plt.title('1-NN')
    plt.show()

    # 3-NN decision regions
    plot_nearest_neighbor(3, data, False)

    plt.title('3-NN')
    plt.show()

    ############### Part (b) ###############
    # 1-NN decision regions
    plot_nearest_neighbor(1, data, True)

    plt.title('Transformed 1-NN')
    plt.show()

    # 3-NN decision regions
    plot_nearest_neighbor(3, data, True)

    plt.title('Transformed 3-NN')
    plt.show()
