import numpy as np
import numpy.linalg as la


class Kernel:
    def rbf(Sigma):
        def k(x, y):
            l2norm = la.norm(x - y)**2
            return np.exp(-l2norm/(2*Sigma*Sigma))
        return k

    def linear():
        def k(x, y):
            return np.inner(x,y)
        return k

    def gaussian(sigma):
        return lambda x, y: \
            np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))