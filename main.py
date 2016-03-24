import numpy as np
import sys
#homemade lovin'
from svm import svm
import parser
import kernels
import random

NUM_COMMITTEES = 2

import matplotlib.pyplot as plt
from randData import twoClasses

#yall know what this bad boy does
def main():

    X = parser.getNumpyArray("TrainX.npy")
    Y = parser.getNumpyArray("TrainY.npy")
    #(X,Y) = twoClasses(600,1,2)
    #np.set_printoptions(threshold=np.nan)
    #print(X)

    x1 = X[:500]
    y1 = Y[:500]
    x2 = X[:500]
    y2 = Y[:500]

    # xs = np.array_split(xM,2)
    # ys = np.array_split(yM,2)


    # Y =ys[0]
    # X =xs[0]

    num_samples = 500
    num_features = 45

    # samples = np.matrix(np.random.normal(size=num_samples * num_features).reshape(num_samples, num_features))
    # labels = 2 * (samples.sum(axis=1) > 0) - 1.0

    # t = svm(samples,labels,kernels.linear())


    # s = t.predict(samples[1][0])
    # print(s)
    # print(labels[1][0])
    t = svm(x1,parser.adjustLabels(y1,1),kernels.rbf(2))
    #print(parser.adjustLabels(y1,1))

    #w = t.getWeights()
    #for i in w:
    #    print(i)

    percentage=0
    for i in range(500):
        percentage += (t.predict(x2[i])>0 and y2[i]==1) or (t.predict(x2[i])<0 and y2[i]!=1)
        #print(t.predict(x2[i],t.getB()))
        #print(y2[i])
    print(percentage/500)



# adjusts the y labels to 1 for curr and -1 for not current
# \param xFileName: filename for the N training vectors
# \param yFilename: filename for the N lables for the training vectors
# \param Kernel:  the kernel function to user on the data
# \return: list of trained svms
def trainSVMs(xFileName, yFIleName, Kernel):
    svms = []

    Y = parser.getNumpyArray(yFIleName)
    X = parser.getNumpyArray(xFileName)

    classToTrain = np.unique(Y.ravel())

    for currClass in classToTrain:
        svms.append(svm(X,parser.adjustLabels(Y,currClass),Kernel))

    return svms


def trainBootstrap():
    numCrossValidationGroups = 5;
    svms = []
    Y = parser.getNumpyArray("TrainY.npy")
    X = parser.getNumpyArray("TrainX.npy")

    # get each class label
    classesToTrain = np.unique(Y.ravel())

    # bootstrap for each class
    for i, currClass in enumerate(classesToTrain):
        classSvms = []

        # adjust Y to be of form not class, class (-1, 1)
        newY = parser.adjustLabels(Y, currClass)

        # bootstrap each group 7 times.  Use 500 each time
        for j in range(0, NUM_COMMITTEES):
            print("Training class %d.  In iteration %d", currClass, j)
            # shuffle arrays together to keep points with classifiers correct 
            combined = list(zip(X, newY))
            random.shuffle(combined)
            X[:], newY[:] = zip(*combined)

            # Get training groups: first 500 in group
            trainY = newY[:500]
            trainX = X[:500]
            
            # train group
            classSvms.append(svm(trainX, parser.adjustLabels(trainY, currClass), kernels.linear()))

        svms.append(classSvms)

    return svms

# takes pre-trained svms and runs points through them, reporting statistics
def predictBootstrap(svms):
    Y = parser.getNumpyArray("TrainY.npy")
    X = parser.getNumpyArray("TrainX.npy")

    # shuffle arrays together to keep points with classifiers correct 
    combined = list(zip(X, Y))
    random.shuffle(combined)
    X[:], Y[:] = zip(*combined)
    X = X[:1]
    classifiers = np.unique(Y.ravel())

    for i, point in enumerate(X):
        print(point)
        # for each classifier j
        count = []
        for j, classifier in enumerate(classifiers):
            count.append(0)
            if(not success):
                # send to each svm to be tested for commitee vote
                for k in range(0,NUM_COMMITTEES):
                    # update count for each commitee vote in this class
                    print(svms[j][k].predict(point))
                    if(np.sign(svms[j][k].predict(point)) > 0):
                        count[j] += 1
        print("Count array for each classifier: ", count)
        # Get classification of point
        


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








