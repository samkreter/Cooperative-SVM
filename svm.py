import numpy as np
from scipy.spatial.distance import pdist, squareform

class svm():
    def __init__(self, X,Y,Kernel):
        self._x = X
        self._y = Y
        self._a = None
        self._b = None
        self._supportVectors = None
        self._sVLables = None
        self._supportWeights = None
        self._kernel = Kernel
        self.train()

    def train(self):
        K = self._gramMatrixGaussian(1)
        T = self._gramMatrix()
        print(K[1][1])
        print("################################")
        print(T[1][1])
        exit(-1)


    # eqn 7.13
    def predict(self, x, b):
        summation = 0;
        for n, x_n in enumerate(self._x):
            summation += self._a[n] * self._y[n] * self._kernel(x, x_n)
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

    # # Stores in self._a to be used in predict function
    # def _getA():
