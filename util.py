import sys
import inspect
import random
import numpy as np
import matplotlib.pyplot as plt


def raiseNotDefined():
    print("Method not implemented: %s" % inspect.stack()[1][3])
    sys.exit(1)

def plotDataset(X):
    plt.plot(X[:, 0], X[:, 1], 'bx')


def plotDatasetClusters(X, mu, z):
    colors = np.array(['b', 'r', 'm', 'k', 'g', 'c', 'y', 'b', 'r', 'm', 'k', 'g', 'c', 'y', 'b', 'r',
                    'm', 'k', 'g', 'c', 'y', 'b', 'r', 'm', 'k', 'g', 'c', 'y', 'b', 'r', 'm', 'k', 'g', 'c', 'y'])
    plt.plot(X[:, 0], X[:, 1], 'w.')
    for k in range(mu.shape[0]):
        plt.plot(X[z == k, 0], X[z == k, 1], colors[k] + '.')
        plt.plot(np.array([mu[k, 0]]), np.array([mu[k, 1]]), colors[k] + 'x')


def sqrtm(M):
    (U, S, VT) = np.linalg.svd(M)
    D = np.diag(np.sqrt(S))
    return np.dot(np.dot(U, D), VT)