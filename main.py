import numpy as np

#homemade lovin'
from svm import svm
import parser
import kernels


currSVMTrainClass = 1
adjustedLables = parser.adjustLabels("TrainY.npy",currSVMTrainClass)

Y = parser.getNumpeeArray("TrainY.npy")
X = parser.getNumpeeArray("TrainX.npy")

def linear(x, y):
    return np.inner(x, y)


t = svm(X,Y,linear)

