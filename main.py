import numpy as np
import sys
#homemade lovin'
from svm import svm
import parser
import kernels
import random

import matplotlib.pyplot as plt
from randData import twoClasses
from subset import randSubset

def main():

    X = parser.getNumpyArray("TrainX.npy")
    Y = parser.getNumpyArray("TrainY.npy")
    #(X,Y) = twoClasses(600,1,2)

    # Shuffle the data
    (Xshuf,Yshuf) = randSubset(X,Y,1656)

    # PARAMETERS
    trainingSampleSize = 1000
    numBootstraps = 1
    bootstrapSampleSize = 0.5 * trainingSampleSize
    testSize = 300
    minConfidence = 0
    C = 10
    minSupportVector = 0.05
    RBF_sigma = 2
    kernel = kernels.rbf(RBF_sigma)

    # Training Data
    x1 = Xshuf[:trainingSampleSize]
    y1 = Yshuf[:trainingSampleSize]

    # Test Data
    x2 = Xshuf[trainingSampleSize : trainingSampleSize + testSize]
    y2 = Yshuf[trainingSampleSize : trainingSampleSize + testSize]

    # Train It Up
    SVMs = trainBootstrapSVMs(x1,y1,kernel,bootstrapSampleSize,numBootstraps,minSupportVector,C)

    # Test the SVMs
    percentage = randomTest(SVMs,x2,y2,testSize,minConfidence)
    print(percentage)


def trainBootstrapSVMs(X,Y,kernel,samps,numCommitteeMembers,minSupportVector,C):
    svms = []

    # get each class label
    classesToTrain = np.unique(Y.ravel())

    # bootstrap for each class
    for i, currClass in enumerate(classesToTrain):
        classSvms = []

        # adjust Y to be of form not class, class; (-1, 1)
        newY = parser.adjustLabels(Y, currClass)

        # bootstrap each group 
        for j in range(0, numCommitteeMembers):
          
            # shuffle arrays together to keep points with classifiers correct 
            #combined = list(zip(X, newY))
            #random.shuffle(combined)
            #newX[:], newY[:] = zip(*combined)

            # Get training groups
            trainY = newY[:samps]
            trainX = X[:samps]
            
            # train group
            classSvms.append(svm(trainX, trainY, C, kernel, minSupportVector))

        svms.append(classSvms)

    return svms


def predictMeanBootstrap(svms,minConfidence,x):
    numClasses, numMembers = np.array(svms).shape
    means = []

    print()

    for i,committee in enumerate(svms):
        means.append(0)

        for member in committee:
            means[i] += member.predict(x)

        means[i] /= numMembers 

    #print(means)

    if max(means) < minConfidence:
        return -1
    else:
        return np.argmax(means) + 1


def predictVarWeightedBootstrap(svms,sigma,minConfidence,x):
    numClasses, numMembers = np.array(svms).shape
    scores = []

    for i,committee in enumerate(svms):
        scores.append(0)
        committeeResults = []

        for member in committee:
            committeeResults.append(member.predict(x))

        scores[i] = np.mean(committeeResults) - sigma*np.std(committeeResults)

    if max(scores) < minConfidence:
        return -1
    else:
        return np.argmax(scores) + 1

def randomTest(svms,X,Y,numTests,minConfidence):

    Xshuf,Yshuf = randSubset(X,Y,numTests)
    correct = 0

    for xn,yn in zip(Xshuf,Yshuf):
        if predictMeanBootstrap(svms,minConfidence,xn) == yn:
            correct += 1
        print(predictMeanBootstrap(svms,minConfidence,xn),yn)

    return correct / numTests


def trainAndStoreSvms(fname):
    svms = trainBootstrap()
    # classifier
    for i, svmSet in enumerate(svms):
        # iteration
        for j, svm in enumerate(svmSet):
            svm.writeSelfToFile(fname, i, j)
    return svms


def loadSvmsFromFile(fname, numClasses, numIterations):
    svms = [];
    for i in range(0, numClasses):
        classSvms = []
        for j in range(0, numIterations):
            currSvm = svm(None, None, kernels.rbf(2), False)
            classSvms.append(currSvm)
            currSvm.loadSelfFromFiles(fname, i, j)
        svms.append(classSvms)
    return svms

if __name__ == '__main__':
    if(len(sys.argv) >= 2 and sys.argv[1] == '1'):
        svms = trainBootstrap()
        predictBootstrap(svms)
    elif(len(sys.argv) >= 2 and sys.argv[1] == '2'):
        svms = trainAndStoreSvms("trainedSVMData/test")
        predictBootstrap(svms)
    elif(len(sys.argv) >= 2 and sys.argv[1] == '3'):
        svms = loadSvmsFromFile("trainedSVMData/test", 8, 7)
        predictBootstrap(svms)
    else:
        main()








