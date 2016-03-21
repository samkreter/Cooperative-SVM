import numpy as np

#homemade lovin'
from svm import svm
import parser
import kernels

Y = parser.getNumpyArray("TrainY.npy")
X = parser.getNumpyArray("TrainX.npy")