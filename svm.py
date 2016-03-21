import numpy as np
from scipy.spatial.distance import pdist, squareform
import cvxopt

class svm():
    def __init__(self, X,Y,Kernel):
        self._x = X
        self._y = Y
        self._c = 1
        self._b = None
        self._supportVectors = None
        self._supportLables = None
        self._supportWeights = None
        self._kernel = Kernel
        self.train()

    def train(self):
        A = self._compute_multipliers(self._x,self._y)

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

        # -a_i \leq 0
        # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))

        print(A.size)
        #exit(-1)
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        return np.ravel(solution['x'])


    # eqn 7.13
    def predict(self, x, b):
        if(b == None):
            b = self._b

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

    # # Stores in self._a to be used in predict function
    # def _getA():
