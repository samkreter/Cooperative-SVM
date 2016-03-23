import numpy as np
import sys
#homemade lovin'
from svm import svm
import parser
import kernels
import random


#yall know what this bad boy does
def main():
    Y = parser.getNumpyArray("TrainY.npy")
    X = parser.getNumpyArray("TrainX.npy")

    num_samples = 1655
    num_features = 45

    samples = np.matrix(np.random.normal(size=num_samples * num_features).reshape(num_samples, num_features))
    labels = 2 * (samples.sum(axis=1) > 0) - 1.0

    t = svm(X,parser.adjustLabels(Y,1),kernels.linear())

    s = t.predict(samples[0][0])
    print(s)
    print(labels[0][0])
    #t = svm(X,parser.adjustLabels(Y,1),kernels.linear())
    #print(t.predict(X[0]))


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


def bootstrap():
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
        for j in range(0, 7):
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


def predict(){
    
}


if __name__ == '__main__':
    if(len(sys.argv) >= 2 and sys.argv[1] == '1'):
        bootstrap()
    else:
        main()




