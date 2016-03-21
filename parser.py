import numpy as np
import sys
import string

def getNumpyArray(filename):
    return np.load(filename)

def write_numpy_array_to_txt(filename, newFileName):
	file = open(newFileName, "w")
	npArray = getNumpyArray(filename)
	for row in npArray:
		file.write( "\n" + np.array_str(row) + ",")
	file.close()