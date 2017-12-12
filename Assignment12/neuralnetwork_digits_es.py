import sys
import time
import matplotlib.pyplot as plt
import numpy as np


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
        x = np.reshape(np.array([1, intensities[i], overall_symmetries[i]]), (3, 1))

        if (input[i][0] == 1.0):
            data.append((x, 1))
        else:
            data.append((x, -1))
        
    return data



# data in form [x, y] where x = [1, intensity, symmetry]
def normalize_data(data):
    min_intensity = min(a.item(1) for (a,b) in data)
    max_intensity = max(a.item(1) for (a,b) in data)

    min_symmetry = min(a.item(2) for (a,b) in data)
    max_symmetry = max(a.item(2) for (a,b) in data)


    for i in range(len(data)):
        intensity = data[i][0].item(1)
        data[i][0][1] =\
            2. * (intensity - min_intensity) / (max_intensity - min_intensity) - 1         

        symmetry = data[i][0].item(2)
        data[i][0][2] =\
            2. * (symmetry - min_symmetry) / (max_symmetry - min_symmetry) - 1


def output_transform(s):
    """
        returns output node transformation
    """
    return s


def output_transform_prime(s):
    """
        returns derivative of output node transformation
    """
    return 1

def theta(s):
    """
        returns transform for forward propagation
    """
    return np.tanh(s)

def theta_prime(xl):
    """
        returns inverse of transform
    """
    return np.subtract(1, np.square(xl))


def forward_propagation(data_point, num_layers, weights):
    """
        data_point is np ndarray of form [1, x1, x2], dimensions (3,1)
    """
    x = [None] * num_layers
    s = [None] * num_layers

    x[0] = data_point
    s[0] = 0

    # transform all middle layers with theta
    # add dummy 1 node to x
    for l in range(1, num_layers-1):
        s[l] = np.dot(np.transpose(weights[l]), x[l-1])
        x[l] = np.concatenate((np.ones((1, 1)), theta(s[l])))

    # transform output node with output transform
    # also don't concatenate dummy 1 node to output
    s[-1] = np.dot(np.transpose(weights[-1]), x[-2])
    x[-1] = output_transform(s[-1])

    return x


def backpropagation(data_point, num_layers, weights, x):
    # delta[0] will be empty
    delta = [None] * num_layers

    # first delta (for layer L) uses derivative of output transform, not theta
    delta[-1] = 2*(x[-1]-data_point[1])*output_transform_prime(x[-1])

    for l in range(num_layers-2, 0, -1):
        delta[l] = np.multiply(theta_prime(x[l]), weights[l+1]*delta[l+1])

    # delete the first element (placeholder)
    for l in range(num_layers-2, 0, -1):
        delta[l] = delta[l][1:]

    return delta


def propagate_and_error(data_point, num_layers, weights, N):
    x = forward_propagation(data_point[0], num_layers, weights)
    delta = backpropagation(data_point, num_layers, weights, x)

    error = float(((1/(4*N))*(x[-1]-data_point[1])**2)[0])

    return x, delta, error
    

def get_error_gradient(data, num_layers, weights):
    error = 0.0
    G = [0]*num_layers

    # G[0] empty, initialize the rest
    for layer in range(1, num_layers):
        G[layer] = np.dot(0, weights[layer])

    # calculate the error and gradient
    for data_point in data:
        x, delta, e = propagate_and_error(data_point, num_layers, weights, len(data))
        error += e

        # find gradient for each layer
        for layer in range(1, num_layers):
            G_l_point = np.dot(x[layer-1], np.transpose(delta[layer]))
            G[layer] += (1/len(data))*G_l_point

    return error, G


def neural_network(data, val_data, num_layers, weights, num_iterations, ax):
    N = float(len(data))
    errors = [0.] * num_iterations
    errors[0] = sys.maxsize
    eta = 0.1
    alpha = 1.05
    beta = 0.7
    no_improvement_ct = 0.0

    # find initial weights' gradient, error
    errors[0], gradient = get_error_gradient(data, num_layers, weights)
    
    # calculate initial validation error
    val_error = 0.0
    for point in val_data:
        _, _, ve = propagate_and_error(point, num_layers, weights, len(val_data))
        val_error += ve

    # loop to improve weights
    for iteration in range(num_iterations-1):
        # find next potential weights' gradient, error
        potential_weights = [w-eta*g for w, g in zip(weights, gradient)]
        potential_error, potential_gradient = get_error_gradient(data, num_layers, potential_weights)

        # check if step good - if good, update weights and stuff
        if potential_error < errors[iteration]:
            weights = potential_weights
            eta = alpha*eta
            gradient = potential_gradient
            errors[iteration+1] = potential_error

            # calculate new validation error
            new_val_error = 0.0
            for point in val_data:
                _, _, ve = propagate_and_error(point, num_layers, weights, len(val_data))
                new_val_error += ve

            if new_val_error < val_error:
                no_improvement_ct = 0
            else:
                no_improvement_ct += 1
            val_error = new_val_error                
        else:
            errors[iteration+1] = errors[iteration]
            eta = beta*eta
            no_improvement_ct += 1

        # if haven't improved validation error in 50 iterations, stop early
        if no_improvement_ct >= 10:
            break

    ax[0].plot(np.arange(num_iterations), errors)

    return weights


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

    training_data = data[:250]
    val_data = data[250:300]    
    testing_data = data[300:]

    # one hidden layer
    num_layers = 3
    # two nodes in hidden layer
    M = 10

    # create weights
    weights = [0]*num_layers
    weights[1] = np.ndarray((3, 10))
    weights[1].fill(0.1)
    weights[2] = np.ndarray((11, 1))
    weights[2].fill(0.1)

    f, ax = plt.subplots(2)
    # # plot training data
    # for point in testing_data:
    #     if point[1] == 1:
    #         ax[1].plot(point[0][1], point[0][2], 'bo')
    #     else:
    #         ax[1].plot(point[0][1], point[0][2], 'rx')

    t = time.time()
    opt_weights = neural_network(training_data, val_data, num_layers, weights, 100000, ax)
    print("TIME: ", time.time() - t)

    # plot decision boundaries
    boundary_points = list()
    for i in range(-100, 100, 2):
        for j in range(-100, 100, 2):
            boundary_points.append((np.reshape(np.array([1, i/100.0, j/100.0]), (3,1)), 0))

    for point in boundary_points:
        x = forward_propagation(point[0], num_layers, opt_weights)

        if np.sign(x[-1]) == 1:
            ax[1].plot(point[0][1], point[0][2], 'bx', alpha=0.08)
        else:
            ax[1].plot(point[0][1], point[0][2], 'rx', alpha=0.08)


    plt.show()

# 1k iters: 11 seconds
# 10k iters: 116 seconds
# 100k iters: 1126 mins