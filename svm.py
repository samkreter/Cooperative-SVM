import numpy as np
from scipy.spatial.distance import pdist, squareform

class svm():
    def __init__(self, X,Y,Kernel):
        self._x = X
        self._y = Y
        self._b = None
        self._supportVectors = None
        self._supportLables = None
        self._supportWeights = None
        self._kernel = Kernel
        self.train()

    def train(self):
        K = self._gramMatrixGaussian(1)
        # A = _getA

        #cool trick I got from tullo with numpy arrays, I'm definitly useing
        # this alot
        #It returns true for all indeces greater than 0 and false for less
        #if a=0 for an element that means it is not a support vector
        supportIndexes = A > 0

        #now we can use that to use only the vectors that we need
        self._supportVectors = self._x[supportIndexes]
        self._supportWeights = A[supportIndexes]
        self._supportLables = self._y[supportIndexes]

        #eqn 7.18
        #using zip trick for the labes and vectors from the tullo blog reference [3]
        self._b = np.mean(tn - self.predict(xn,0) for (tn,xn) in zip(self.supportLables,supportVectors))




    # eqn 7.13
    def predict(self, x, b):
        summation = 0;
        for n, x_n in enumerate(self.supportVectors):
            summation += self._supportWeights[n] * self._supportLables[n] * self._kernel(x, x_n)
        return summation + b

    # def langranging_multipliers():
    #     getGramMatrix()
    #     getP
    #     getQ
    #     getA()

    #comute the gram matrix super fast but jsut for the guassian/RBF
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
        np.mean()

    # def _getP():


    # def _getQ():

    # Stores in self._a to be used in predict function
    # def _getA():

