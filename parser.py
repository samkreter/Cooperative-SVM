import numpy as np
import sys
import string

def getNumpyArray(filename):
    return np.load(filename)

def write_numpy_array_to_txt(filename, newFileName):
    # print(filename)
    # print
	file = open(newFileName, "w")
	npArray = getNumpyArray(filename)
	for row in npArray:
		file.write(np.array_str(row) + ",\n")
	file.close()

def write_numpy_array_to_npy(newFileName, npArray):
    np.save(newFileName, npArray)

def adjustLabels(originalLabels, PrimaryClass):
    newLabels = []
    for label in originalLabels:
        if label == PrimaryClass:
            newLabels.append([1*1.])
        else:
            newLabels.append([-1*1.])
    return np.array(newLabels)