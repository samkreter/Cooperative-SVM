#Implementing an SVM for Handwriting Recognition

##Running the code with default settings

This section will show how to run the code with the present parameters.

##Parameters

trainingSampleSize: Training sample size (Default value: 1000)

numBootstraps: Number of bootstrap samples per class (Default value: 1)

bootstrapSampleSize: Bootstrap sample size (Default value: math.floor(0.7 * trainingSampleSize))

testSize: Number of points in the test set (Default value: 500)

minConfidence: Minimum confidence needed for a class prediction (Default value: 0)

C: Tradeoff parameter for the slack variables (Default value: 1)

minSupportVector: Minimum value for point to be considered a support vector (Default value: 0.1)

RBF_sigma: The variance of the kernel (Default value: 1)

kernel: The kernel used in the calculations (Default value: kernels.rbf(RBF_sigma))