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


# change raw text input so it is only storing 1s and 5s, 
# return a list in form [number, greyscale values]
def clean_input(data):
    cleanedData = list()

    i = 0
    while i < len(data):
        # only interested in 1 or 5 & corresponding values
        if (data[i] != 1. and data[i] != 5. ):
            i += 257
            continue
        
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
def create_data(input, a):
    data = list()
    
    intensities =  get_intensities(input)
    horizontal_symmetries = np.array(get_horizontal_symmetries(input))
    vertical_symmetries = np.array(get_vertical_symmetries(input))
    
    overall_symmetries = horizontal_symmetries+vertical_symmetries

    # input in form of array[[num, grayscal vals]
    for i in range(len(input)):
        x1 = intensities[i]
        x2 = overall_symmetries[i]
        x = np.array([1, x1, x2, x1**2, x1*x2, x2**2, x1**3, (x1**2)*x2, x1*(x2**2), x2**3])

        if (input[i][0] == 1.0):
            data.append((x, 1))
            a.plot(intensities[i], overall_symmetries[i], 'bo', mew=3, ms=10, fillstyle='none', markeredgecolor='blue')    
        else:
            data.append((x, -1))
            a.plot(intensities[i], overall_symmetries[i], 'rx', mew=3, ms=10)
    
    return data

# determine if point classified correctly
def goodH(w, point):
    result = np.dot(w, point[0])

    if (np.sign(result) == np.sign(point[1])):
        return True

    return False


#run PLA algorithm
def pla_once(w, data, n):
    # search for misclassified point
    for i in range(n):
        # if point is classified correcly, keep looking
        if (goodH(w, data[i]) == True):
            continue
        else:
            # if point misclassified, use update rule and stop looking
            w = w + data[i][1] * data[i][0]
            break
    
    return w


# find a better weight through linear regression
def linear_regression(data, n):
    x_a = np.zeros(shape=(n,10))
    y_a = np.zeros(shape=(n,1))

    for i in range(n):
        x_a[i] = data[i][0]
        y_a[i] = data[i][1]
        
    x = np.asmatrix(x_a)
    y = np.asmatrix(y_a)

    x_transpose = x.transpose() 

    psuedo_inverse_x = np.linalg.inv(x_transpose * x) * x_transpose

    opt = psuedo_inverse_x * y

    opt_array = np.squeeze(np.asarray(opt))

    return opt_array


# get the number of misclassified points
def get_misclassified(w, data, n):
    err_count = 0

    # for each point in the data set, check if point misclassified
    for i in range(n):
        # if point misclassified, increment error count
        if (goodH(w, data[i]) == False):
            err_count += 1

    return err_count


# check if new set of weights classifies point better than old one
def should_change(w_hat, w_new, data, n):
    # see how many points old weights misclassify
    old_errors = get_misclassified(w_hat, data, n)
    # see how many points new weights misclassify
    new_errors = get_misclassified(w_new, data, n)
    
    if (new_errors < old_errors):
        return True

    return False


# run pocket algorithm
def pocket(data, n, max_iters):
    # get a good starting weights by using linear regression
    # w = [0,0,0]
    w = linear_regression(data, n)
    w_hat = w
    

    for i in range(max_iters):
        # run a single PLA iteration to get some new weights
        w = pla_once(w, data, n)

        # set w_hat to better of two weights
        if (should_change(w_hat, w, data, n) == True):
            w_hat = w

    return w_hat


def main():
    max_iters = 100

    fig, axarr = plt.subplots(2, sharex=True)

    # get all data from files
    train = get_input('ZipDigits.train')
    test = get_input('ZipDigits.test')

    # create our data and test input arrays in form [num, grayscaleNums]
    training_input = clean_input(train)
    testing_input = clean_input(test)

    # create training and testing data
    training_data = create_data(training_input, axarr[0]) 
    testing_data = create_data(testing_input, axarr[1])

    # run pocket algo on training, testing data
    w_train = pocket(training_data, len(training_data), max_iters)
    # w_test = pocket(testing_data, len(testing_data), max_iters)

    # error calculated by total number of misclassified points / total number of points
    e_in = float(get_misclassified(w_train, training_data, len(training_data))) / len(training_data)
    e_test = float(get_misclassified(w_train, testing_data, len(testing_data)))  / len(testing_data)

    print "Ein = ", e_in
    print "Etest = ", e_test

    error_bar_train = math.sqrt((8. / len(training_data)) * np.log((4 * ((2 * len(training_data))**10 + 1)) / 0.05))
    error_bar_test = math.sqrt(1./(2*len(testing_data)) * np.log(2./0.05))
    
    e_out_train = e_in + error_bar_train
    e_out_test = e_test + error_bar_test

    print "Eout_train: ", e_out_train
    print "Eout_test: ", e_out_test 

    # plot our optimal weight
    x1 = np.linspace(-250, 50, 300)
    x2 = np.linspace(-250, 0, 300)
    x1, x2 = np.meshgrid(x1, x2)


    axarr[0].contour(x1, x2, w_train[0] + w_train[1]*x1 + w_train[2]*x2 + w_train[3]*x1**2 + w_train[4]*x1*x2 + w_train[5]*x2**2 + w_train[6]*x1**3 + w_train[7]*x1**2*x2 + w_train[8]*x1*x2**2 + w_train[9]*x2**3, [0])    
    axarr[1].contour(x1, x2, w_train[0] + w_train[1]*x1 + w_train[2]*x2 + w_train[3]*x1**2 + w_train[4]*x1*x2 + w_train[5]*x2**2 + w_train[6]*x1**3 + w_train[7]*x1**2*x2 + w_train[8]*x1*x2**2 + w_train[9]*x2**3, [0])
    
    axarr[0].set_title('Training Data')
    axarr[1].set_title('Testing Data')
    plt.xlabel('intensity')
    plt.ylabel('symmetry')
    plt.show()




if __name__ == '__main__':
    main()