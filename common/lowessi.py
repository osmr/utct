from math import ceil
import numpy as np
from scipy import linalg


class Lowessi(object):
    """
    This class implements the incremetal Lowess function for nonparametric regression.
    This realization is based on the Lowess function of Michael Eickenberg and Fabian Pedregosa.
    """

    @staticmethod
    def preprocess(x, f=2. / 3.):
        """
        Preprocessing for the incremetal Lowess function.

        Parameters:
        ----------
        x : np.array of float
            x-values array
        f : float
            smoothing span parameter

        Returns:
        ----------
        xx : np.array of float
            auxiliary array of x^2
        w : np.array of float
            auxiliary array of weights
        """
        xx = x * x
        n = len(x)
        r = int(ceil(f * n))
        h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
        w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
        w = (1 - w ** 3) ** 3
        return xx, w


    @staticmethod
    def update(y, x, xx, w, iter=3):
        """
        Incremental update function for Lowess.

        Parameters:
        ----------
        y : np.array of float
            y-values array
        x : np.array of float
            x-values array
        xx : np.array of float
            auxiliary array of x^2
        w : np.array of float
            auxiliary array of weights
        iter : int
            number of robustifying iterations

        Returns:
        ----------
        yest : np.array of float
            smoothed y-values
        """
        yx = y * x
        n = len(x)
        yest = np.zeros(n)
        delta = np.ones(n)
        for iteration in range(iter):
            for i in range(n):
                weights = delta * w[:, i]
                a11 = np.sum(weights)
                a12 = np.sum(weights * x)
                a21 = a12
                a22 = np.sum(weights * xx)
                b1 = np.sum(weights * y)
                b2 = np.sum(weights * yx)
                det = a11 * a22 - a12 * a21
                if abs(det) > 1e-8:
                    yest[i] = ((a22 * b1 - a12 * b2) +
                               (a11 * b2 - a21 * b1) * x[i]) / det
                else:
                    yest[i] = yest[i - 1] if i > 0 else 0.0

            residuals = y - yest
            s = np.median(np.abs(residuals))
            delta = np.clip(residuals / (6.0 * s), -1, 1)
            delta = (1 - delta ** 2) ** 2

        return yest


def main():

    import math
    n = 100
    x = np.linspace(0, 2 * math.pi, n)
    y = np.sin(x) + 0.3 * np.random.randn(n)

    #x = np.linspace(0., 1.0, n)
    #y = 1.0 + 0.0001*np.random.randn(n)

    f = 0.5

    xx, w = Lowessi.preprocess(x, f=f)
    yest = Lowessi.update(y, x, xx, w, iter=3)

    import pylab as pl
    pl.clf()
    pl.plot(x, y, label='y noisy')
    pl.plot(x, yest, label='y pred')
    pl.legend()
    pl.show()
    a = 1


if __name__ == '__main__':
    main()
