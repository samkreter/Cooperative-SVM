import numpy as np

#homemade lovin'
from svm import svm
import parser
import kernels


#yall know what this bad boy does
def main():
    Y = parser.getNumpyArray("TrainY.npy")
    X = parser.getNumpyArray("TrainX.npy")

    num_samples = 160
    num_features = 2

    samples = np.matrix(np.random.normal(size=num_samples * num_features).reshape(num_samples, num_features))
    labels = 2 * (samples.sum(axis=1) > 0) - 1.0

    t = svm(samples,labels,kernels.linear())




    s = t.predict(samples[0][0])
    print(s)
    print(labels[0][0])
    #t = svm(X,parser.adjustLabels(Y,1),kernels.linear())
    #print(t.predict(X[0]))


# adjusts the y labels to 1 for curr and -1 for not current
# \param xFileName: filename for the N training vectors
# \param yFilename: filename for the N lables for the training vectors
# \param Kernel:  the kernel function to user on the data
# \return: list of trained svms
def trainSVMs(xFileName, yFIleName, Kernel):
    svms = []

    Y = parser.getNumpyArray(yFIleName)
    X = parser.getNumpyArray(xFileName)

    classToTrain = np.unique(Y.ravel())

    for currClass in classToTrain:
        svms.append(svm(X,parser.adjustLabels(Y,currClass),Kernel))

    return svms


def crossValidate():
    numCrossValidationGroups = 5;

    svms = [][]

	Y = parser.getNumpyArray("TrainY.npy")
	X = parser.getNumpyArray("TrainX.npy")

	# get each class label
	classToTrain = np.unique(Y.ravel())
	
	numEachClass = []
	for i, classifier in enumerate(classToTrain):
		numEachClass[i] = y.count(classifier)

	# cross-validate for each class
	for i, currClass in enumerate(classToTrain):
		classSvms = []
		# shuffle arrays together to keep points with classifiers correct 
		combined = zip(X, Y)
		random.shuffle(combined)
		X[:], Y[:] = zip(*combined)
		# adjust Y to be of form not class, class (-1, 1)
		newY = parser.adjustLabels(Y,currClass)

		# split each class's points into groups to cross validate.  Exclude 1/5 each time
		for j in range(numEachClass[i]):
			trainY = numpy.delete()
			svms.append(svm(X,parser.adjustLabels(Y,currClass),Kernel))


	    svms.append(svm(X,parser.adjustLabels(Y,currClass),Kernel))

    return svms


if __name__ == '__main__':
	if(argv[1] == true){
		crossValidate()
	}
	else{
    	main()
    }




