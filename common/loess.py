# Author Michael Eickenberg <michael.eickenberg@nsup.org>, Fabian Pedregosa
# Coded in 2012, another era, pure python, no guarantees for 1000%
# correctness or speed

"""
This module implements the Lowess function for nonparametric regression.
 
Functions:
lowess Fit a smooth nonparametric regression curve to a scatterplot.
 
For more information, see
 
William S. Cleveland: "Robust locally weighted regression and smoothing
scatterplots", Journal of the American Statistical Association, December 1979,
volume 74, number 368, pp. 829-836.
 
William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
approach to regression analysis by local fitting", Journal of the American
Statistical Association, September 1988, volume 83, number 403, pp. 596-610.
"""

from math import ceil
import numpy as np
from scipy import linalg


def lowess(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
 
Lowess smoother: Robust locally weighted regression.
The lowess function fits a nonparametric regression curve to a scatterplot.
The arrays x and y contain an equal number of elements; each pair
(x[i], y[i]) defines a data point in the scatterplot. The function returns
the estimated (smooth) values of y.
 
The smoothing span is given by f. A larger value for f will result in a
smoother curve. The number of robustifying iterations is given by iter. The
function will run faster with a smaller number of iterations."""
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w**3)**3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta**2)**2

    return yest


def lowess2_preprocess(x, f=2. / 3.):
    xx = x * x
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    return xx, w


def lowess2(y, x, xx, w, iter=3):

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


if __name__ == '__main__':
    import math
    n = 100
    x = np.linspace(0, 2 * math.pi, n)
    y = np.sin(x) + 0.3 * np.random.randn(n)

    #x = np.linspace(0., 1.0, n)
    #y = 1.0 + 0.0001*np.random.randn(n)

    f = 0.5

    #yest = lowess(x, y, f=f, iter=3)
    xx, w = lowess2_preprocess(x, f=f)
    yest = lowess2(y, x, xx, w, iter=3)

    import pylab as pl
    pl.clf()
    pl.plot(x, y, label='y noisy')
    pl.plot(x, yest, label='y pred')
    pl.legend()
    pl.show()
    a = 1
