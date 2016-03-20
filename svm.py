import numpy as np


class svm():

    def __init__(self, X,Y,Kernel):
        self._x = X
        self._y = Y
        self._kernel = Kernel
        self.train()


    def train(self):
        K = self._gramMatrix()
        # a = langranging_multipliers()
        # b = getB()


    def predict(self):


    # def langranging_multipliers():
    #     getGramMatrix()
    #     getP
    #     getQ
    #     getA()


    def _gramMatrix(self):
        n_samples, n_features = self._x.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(self._x):
            for j, x_j in enumerate(self._x):
                K[i, j] = self._kernel(x_i, x_j)
        return K
