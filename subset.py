import numpy as np
import math
import random

def randSubset(X,Y,n):
	combined = list(zip(X,Y))
	random.shuffle(combined)
	X[:],Y[:] = zip(*combined)
	X = X[:n]
	Y = Y[:n]

	return (X,Y)
