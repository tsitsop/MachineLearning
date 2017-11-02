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
        
        # # array of greyscale vals
        # info = [0.]*256
        # # each of next 256 values are greyscale so add to info
        # for j in range(256):
        #     info[j] = data[(i+1)+j]

        # cleanedData.append([data[i], info])

        # create a list of all relevant data
        for j in range(256):
            cleanedData.append(data[(i+1)+j])

        i += 257

    return cleanedData

def writeToFile(filename, data):
    f = open(filename, 'w')

    for d in data:
        f.write("%s," % d)

def main():
    # get all data from files
    train = getData('ZipDigits.train')
    test = getData('ZipDigits.test')

    # create our data and test arrays in form [num, grayscaleNums]
    training_data = cleanData(train)
    testing_data = cleanData(test)

    writeToFile('correct.train', training_data)
    writeToFile('correct.test', testing_data)


if __name__ == "__main__":
    main()