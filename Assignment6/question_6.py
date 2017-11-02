import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import quad


def getData(filename):
    f = open(filename)
    data = f.read()

    parsedStringData = data.split(" ")

    del parsedStringData[-1]

    parsedData = map(float, parsedStringData)

    return parsedData

def cleanData(data):
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

def partA(data):
    for i in range(2):
        # get sample digit in matrix form
        imgarray = np.array(data[i][1])
        img = np.reshape(imgarray, (-1, 16))

        # recreate the digit
        plt.imshow(img, cmap='gray_r', interpolation='nearest')
        
        plt.show()

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

        
        
def main():
    data = getData('ZipDigits.train')
    test = getData('ZipDigits.test')



    # create our data and test arrays in form [num, grayscaleNums]
    data = cleanData(data)
    test = cleanData(test)


    data_length = len(data)

    # part A
    # partA(data)

    # Feature 1: average intensity
    #   - bigger intensity = more black pixels
    intensities =  get_intensities(data)
    # Feature 2: symmetry
    #   - closer to 0 = more symmetrical
    horizontal_symmetries = np.array(get_horizontal_symmetries(data))
    vertical_symmetries = np.array(get_vertical_symmetries(data))
    print np.sum(horizontal_symmetries - vertical_symmetries)
    # print np.sum(vertical_symmetries)


    overall_symmetries = horizontal_symmetries+vertical_symmetries
    # # find intensities and symmetries based on digit
    intensities_1s = list()
    symmetries_1s = list()
    intensities_5s = list()
    symmetries_5s = list()
    for i in range(data_length):
        if (data[i][0] == 1.0):
            intensities_1s.append(intensities[i])
            symmetries_1s.append(overall_symmetries[i])
            # symmetries_1s.append(horizontal_symmetries[i])
        else:
            intensities_5s.append(intensities[i])
            symmetries_5s.append(overall_symmetries[i])
            # symmetries_5s.append(horizontal_symmetries[i])



    # # plot our points
    plt.plot(intensities_1s, symmetries_1s, 'bo', mew=3, ms=10, fillstyle='none', markeredgecolor='blue')
    plt.plot(intensities_5s, symmetries_5s, 'rx', mew=3, ms=10)
    
    plt.xlabel('Average Intensity')
    plt.ylabel('Symmetry')
    plt.show()



if __name__ == "__main__":
    main()    




