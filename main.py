import numpy as np

#homemade lovin'
from svm import svm
import parser

Y = parser.getNumpeeArray("TrainY.npy")
X = parser.getNumpeeArray("TrainX.npy")

def linear(x, y):
    return np.inner(x, y)


t = svm(X,Y,linear)