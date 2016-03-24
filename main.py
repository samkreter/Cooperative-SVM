import numpy as np

#homemade lovin'
from svm import svm
import parser
import kernels
import matplotlib.pyplot as plt
from randData import twoClasses
from subset import randSubset

#yall know what this bad boy does
def main():

    X = parser.getNumpyArray("TrainX.npy")
    Y = parser.getNumpyArray("TrainY.npy")
    #(X,Y) = twoClasses(600,1,2)

    (Xshuf,Yshuf) = randSubset(X,Y,1656)

    # PARAMETERS
    sampleSize = 500
    testSize = 300
    targetClass = 4

    # Training Data
    x1 = Xshuf[:sampleSize]
    y1 = Yshuf[:sampleSize]

    # Test Data
    x2 = Xshuf[sampleSize:sampleSize+testSize]
    y2 = Yshuf[sampleSize:sampleSize+testSize]

    # Train It Up
    t = svm(x1,parser.adjustLabels(y1,targetClass),kernels.rbf(2))
    #print(parser.adjustLabels(y1,1))

    correct = 0
    for i in range(testSize):
        correct += (t.predict(x2[i])>0 and y2[i]==1) or (t.predict(x2[i])<0 and y2[i]!=1)
        #print(t.predict(x2[i])
        #print(y2[i])
    print(correct/testSize)


# adjusts the y labels to 1 for curr and -1 for not current
# \param xFileName: filename for the N training vectors
# \param yFilename: filename for the N lables for the training vectors
# \param Kernel:  the kernel function to user on the data
# \return: list of trained svms
def trainSVMs(xFileName,yFIleName,Kernel):
    svms = []

    Y = parser.getNumpyArray(yFIleName)
    X = parser.getNumpyArray(xFileName)

    classToTrain = np.unique(Y.ravel())

    for currClass in classToTrain:
        svms.append(svm(X,parser.adjustLabels(Y,currClass),Kernel))

    return svms


if __name__ == '__main__':
    main()



