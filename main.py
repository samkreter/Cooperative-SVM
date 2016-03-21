import numpy as np

#homemade lovin'
from svm import svm
import parser
import kernels

Y = parser.getNumpeeArray("TrainY.npy")
X = parser.getNumpeeArray("TrainX.npy")