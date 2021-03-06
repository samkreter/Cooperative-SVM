import numpy as np
from scipy.spatial.distance import pdist, squareform
import cvxopt
import parser
import multiplyer

class svm():
    def __init__(self,X,Y,C,Kernel,minSupportVector,train=True):
        self._x = X
        self._y = Y
        self._c = C
        self._b = None
        self._supportVectors = None
        self._supportLabels = None
        self._supportWeights = None
        self._kernel = Kernel
        self._minSupportVector = minSupportVector
        if(train):
            self.train(minSupportVector)

    def train(self,minSupportVector):
        A = self._compute_multipliers(self._x,self._y)
        # K = self._gramMatrix()
        # A = multiplyer.CalculateLagrangeMultipliers(self._y,K,self._c)
        #cool trick I got from tullo with numpy arrays, I'm definitly useing
        # this alot
        #It returns true for all indeces greater than 0 and false for less
        #if a=0 for an element that means it is not a support vector
        supportIndexes = A > self._c * minSupportVector

        #now we can use that to use only the vectors that we need
        self._supportVectors = self._x[supportIndexes]
        self._supportWeights = A[supportIndexes]
        self._supportLabels = self._y[supportIndexes]
        print("Support Vectors:",len(self._supportLabels))


        if len(self._supportLabels) == 0:
            self._b = 0
        else:
            self._b = np.mean([tn - self.predict(xn,0) for (tn,xn) in zip(self._supportLabels,self._supportVectors)])



    def _compute_multipliers(self, X, y):
        n_samples, n_features = X.shape

        K = self._gramMatrix()
        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q,G,h,A,b)

        # Lagrange multipliers
        return np.ravel(solution['x'])


    # eqn 7.13
    def predict(self, x, b=None):
        if(b == None):
            b = self._b

        summation = b;

        if len(self._supportLabels) == 0:
            return summation

        for n, x_n in enumerate(self._supportVectors):
            summation += self._supportWeights[n] * self._supportLabels[n] * self._kernel(x, x_n)
        return summation.item()

    # def langranging_multipliers():
    #     getGramMatrix()
    #     getP
    #     getQ
    #     getA()

    #comute the gram matrix super fast but just for the guassian/RBF
    def _gramMatrixGaussian(self,sigma):
        pairwise_dists = squareform(pdist(self._x, 'euclidean'))
        K = np.exp(pairwise_dists ** 2 / sigma ** 2)
        return K

    def _gramMatrix(self):
        n_samples, n_features = self._x.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(self._x):
            for j, x_j in enumerate(self._x):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def getB(self):
        return self._b;

    def getWeights(self):
        return self._supportWeights

    # def _getP():


    # def _getQ():

    # Stores in self._a to be used in predict function
    # def _getA():

    # Passfilename without extension to be appended for each data piece
    def writeSelfToFile(self, filename, classifier, iteration):
        fname = ""+filename + "_supportVectors_"+str(classifier)+"_"+str(iteration)+".npy"
        parser.write_numpy_array_to_npy(fname, self._supportVectors)

        fname = "{}_supportWeights_{}_{}.npy".format(filename, classifier, iteration)
        parser.write_numpy_array_to_npy(fname, self._supportWeights)

        fname = "{}_supportLabels_{}_{}.npy".format(filename, classifier, iteration)
        parser.write_numpy_array_to_npy(fname, self._supportLabels)

        # fname = "{}_kernel_{}_{}.npy".format(filename, classifier, iteration)
        # parser.write_numpy_array_to_txt2(fname, self._kernel)

        fname = "{}_b_{}_{}.npy".format(filename, classifier, iteration)
        parser.write_numpy_array_to_npy(fname, [self._b])



    def loadSelfFromFiles(self, filename, classifier, iteration):
        fname = ""+filename + "_supportVectors_"+str(classifier)+"_"+str(iteration)+".npy"
        self._supportVectors = parser.getNumpyArray(fname)

        fname = "{}_supportWeights_{}_{}.npy".format(filename, classifier, iteration)
        self._supportWeights = parser.getNumpyArray(fname)

        fname = "{}_supportLabels_{}_{}.npy".format(filename, classifier, iteration)
        self._supportLabels  = parser.getNumpyArray(fname)

        fname = "{}_b_{}_{}.npy".format(filename, classifier, iteration)
        self._b              = parser.getNumpyArray(fname)[0]




