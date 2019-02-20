from numpy import *
from util import *
import util

def pca(X, K):
    '''
    X is an N*D matrix of data (N points in D dimensions)
    K is the desired maximum target dimensionality (K <= min{N,D})

    should return a tuple (P, Z, evals)
    
    where P is the projected data (N*K) where
    the first dimension is the higest variance,
    the second dimension is the second higest variance, etc.

    Z is the projection matrix (D*K) that projects the data into
    the low dimensional space (i.e., P = X * Z).

    and evals, a K dimensional array of eigenvalues (sorted)
    '''
    
    N,D = X.shape
    P, Z, evals = None, None, None 

    # make sure we don't look for too many eigs!
    if K > N:
        K = N
    if K > D:
        K = D

    # first, we need to center the data
    mean = np.mean(X, axis=0)

    M = (X - mean).T.dot((X - mean)) / (np.real(X.shape[0])-1)

    # next, compute eigenvalues of the data variance
    (evals, evecs) = np.linalg.eig(M)
    evals = evals.real
    evecs = evecs.real

    idx = np.argsort(evals)[::-1]  
    eigenValues = evals[idx]
    eigenVectors = evecs[:,idx]
    eig_pairs = [(np.abs(evals[i]), evecs[:,i]) for i in range(len(evals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()

    Z = np.hstack((eig_pairs[i][1].reshape(K,1) for i in range(D)))
   
    P = np.dot(X, Z)
    evals = eigenValues

    return (P, Z, evals)


