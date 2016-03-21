import numpy as np


class svm():

    def __init__(self, X,Y,Kernel):
        self._x = X
        self._y = Y
        self._kernel = Kernel
        self.train()


    def train(self):
        K = self._gramMatrix()


    # eqn 7.13
    def predict(self, x, b):
        summation = 0;
        for n, x_n in enumerate(self._x):
            summation += self._a[n] * self._y[n] * self._kernel(x, x_n)
        return summation + b



    def _gramMatrix(self):
        n_samples, n_features = self._x.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(self._x):
            for j, x_j in enumerate(self._x):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    # def _getP():


    # def _getQ():


    # # Stores in self._a to be used in predict function
    # def _getA():
