import os
import numpy as np
import operator
import cv2


def img2vector(img_path):
    """
    Format the input data image and convert it to a vector
    return: 1*784 number vector
    """
    # Using 0 to read image in grayscale mode
    image = cv2.imread(img_path, 0)
    dataVect = np.zeros((1, 784))
    # extract data from 28*28 image with corresponding img path
    for i in range(28):  # y direction
        for j in range(28):  # x direction
            if int(image[i][j]) != 0:  # if the pixel is not white
                dataVect[0, 28 * i + j] = int(1)  # put 1 into the 1 * 784 matrix
            else:  # if the pixel has color
                # Offsets are used to distribute the entire
                # handwritten numeric data across an array
                dataVect[0, 28 * i + j] = int(image[i][j])  # put 0 into the 1 * 784 matrix
    # print(dataVect)
    return dataVect


def kNN_classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # Calculating Euclidean distance
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}

    # Select k points with the smallest distance
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # Sorting
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print(sortedClassCount)
    return sortedClassCount[0][0]


def handwritingClassficationTest():
    # loading the image data from the file path----start
    hwLabels = []
    path = './imgs_test'
    file_list = os.listdir(path)
    number = 0
    for i in file_list:
        img_path = os.path.join(path, i)
        img_list = os.listdir(img_path)
        number += len(img_list)
    m = number
    num = 0
    trainingMat = np.zeros((m, 784))
    # loading the image data from the file path----end
    for i in file_list:
        img_path = os.path.join(path, i)
        img_list = os.listdir(img_path)
        for j in img_list:
            img_path_one = os.path.join(img_path, j)
            # print(img_path_one)
            trainingMat[num, :] = img2vector(img_path_one)
            num += 1
            hwLabels.append(int(i))

    test_path = './test'
    test_file_list = os.listdir(test_path)
    number_test = 0
    errorCount = 0
    for i in test_file_list:
        img_path = os.path.join(path, i)
        img_list = os.listdir(img_path)
        number_test += len(img_list)
        for j in img_list:
            vectorUnderTest = img2vector(os.path.join(img_path, j))
            classNumStr = int(i)
            classifierResult = kNN_classify0(vectorUnderTest, trainingMat, hwLabels, 3)
            if classifierResult != classNumStr:
                errorCount += 1.0

    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total accuracy is: %f" % (1 - errorCount / float(number_test)))


def test_one_pic(test_pic_path):
    hwLabels = []
    path = './imgs_test'
    file_list = os.listdir(path)  # Loading training data
    number = 0
    for i in file_list:
        img_path = os.path.join(path, i)
        img_list = os.listdir(img_path)
        number += len(img_list)

    m = number
    num = 0
    trainingMat = np.zeros((m, 784))
    # print(file_list)
    for i in file_list:
        img_path = os.path.join(path, i)
        img_list = os.listdir(img_path)
        for j in img_list:
            img_path_one = os.path.join(img_path, j)
            trainingMat[num, :] = img2vector(img_path_one)
            # print(img2vector(img_path_one))
            num += 1
            hwLabels.append(int(i))
    vectorUnderTest = img2vector(test_pic_path)
    classifierResult = kNN_classify0(vectorUnderTest, trainingMat, hwLabels, 3)
    print("\nThe predicted result is: ", classifierResult)


def askUserInput():
    print('Please make selection from following choices:\nPress 1 for KNN accuracy\nPress 2 for picture '
          'identify\nPress 3 to quit the program\n')
    userIn = input("Enter your value:")

    while userIn != "3":
        if userIn == "2":
            print('Please enter the file name, format please see '
                  'below:\n./test/6/6_011.jpg\n./test/1/1_000.jpg\n./test/#/#_###\nIf you want to see available file '
                  'name, please see the picture in test folder\nPress enter to end input')
            fileIn = input("Enter your filename:")
            test_one_pic(fileIn)
        elif userIn == "1":
            print('Please hold...')
            handwritingClassficationTest()  # Test the accuracy of the entire set
        else:
            print('Invalid input! Please try again!\n')
        print('\nPlease make selection from following choices:\nPress 1 for KNN accuracy\nPress 2 for picture '
              'identify\nPress 3 to quit the program\n')
        userIn = input("Enter your value:")
    print("Thank you for using!")


if __name__ == "__main__":
    askUserInput()
