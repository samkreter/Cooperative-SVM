#Implementing an SVM for Handwriting Recognition

##Quick Start

Download the zip file containing the SVM files. Use the train.py function to train and the test.py function to test.

##Installation

###Python Dependencies

numpy >= 1.10.4

cvxopt >= 1.1.8

##Running the code with default settings

Put the input data into the train function in train.py and then use the test function in test.py to test the data.

##Parameters

**trainingSampleSize:** Training sample size (Default value: 1000)

**numBootstraps:** Number of bootstrap samples per class (Default value: 1)

**bootstrapSampleSize:** Bootstrap sample size (Default value: math.floor(0.7 * trainingSampleSize))

**testSize:** Number of points in the test set (Default value: 500)

**minConfidence:** Minimum confidence needed for a class prediction (Default value: 0)

**C:** Tradeoff parameter for the slack variables (Default value: 1)

**minSupportVector:** Minimum value for point to be considered a support vector (Default value: 0.1)

**RBF_sigma:** The variance of the kernel (Default value: 1)

**kernel:** The kernel used in the calculations (Default value: kernels.rbf(RBF_sigma))