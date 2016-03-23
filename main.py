import numpy as np

#homemade lovin'
from svm import svm
import parser
import kernels


#yall know what this bad boy does
def main():

    X = parser.getNumpyArray("TrainX.npy")
    Y = parser.getNumpyArray("TrainY.npy")

    x1 = X[:500]
    y1 = Y[:500]
    x2 = X[500:600]
    y2 = Y[500:600]

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
    t = svm(x1,parser.adjustLabels(y1,1),kernels.gaussian(1))

    for i in range(20):
        print(t.predict(x2[i]))
        print(y2[i])


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



