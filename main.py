import numpy as np

#homemade lovin'
from svm import svm
import parser
import kernels
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

    # num_samples = 160
    # num_features = 2

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



