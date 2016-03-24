Implementing an SVM for Handwriting Recognition


Quick Start

    Download the zip file containing the SVM files. Use the train.py function to train and the test.py function to test.

Installation

    Python Dependencies

        numpy >= 1.10.4

        cvxopt >= 1.1.8

Running the code with default settings

    In order to train the SVM, run the train.py python file with two arguments X_train and Y_train, where X_train is the input training data and Y_train is the desired output vector.

    `python train.py X_train Y_train`

    In order to test the SVM, run the test.py python file with one argument X_test, when the X_test is the input testing data points.

    `python test.py X_test`

    The result of test.py will be a vector with the predicted classification of each input testing data point.

Parameters

    trainingSampleSize: Training sample size (Default value: 1200)

    numBootstraps: Number of bootstrap samples per class (Default value: 1)

    bootstrapSampleSize: Bootstrap sample size (Default value: math.floor(0.7 * trainingSampleSize))

    testSize: Number of points in the test set (Default value: 500)

    minConfidence: Minimum confidence needed for a class prediction (Default value: 0.1)

    C: Tradeoff parameter for the slack variables (Default value: 4)

    minSupportVector: Minimum value for point to be considered a support vector (Default value: 0.1)

    RBF_sigma: The variance of the kernel (Default value: 1)

    kernel: The kernel used in the calculations (Default value: kernels.rbf(RBF_sigma))