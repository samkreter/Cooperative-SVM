import math
import numpy as np
import sys
	
def twoClasses(n,sigma,sep):

	X = np.ones((n,2))
	Y = np.ones(n)

	for i in range(n):
		if i > n/2:
			X[i][0] = -sep+np.random.normal(0,sigma)
			X[i][1] = np.random.normal(0,sigma)
			Y[i] = 0
		else:
			X[i][0] = sep+np.random.normal(0,sigma)
			X[i][1] = sep+np.random.normal(0,sigma)
			Y[i] = 1

	return (X,Y)
