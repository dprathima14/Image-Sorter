import numpy as np
import util
import matplotlib.pyplot as plt

def kmeans(X, mu0, doPlot=True, verbose=True):
    '''
    X is an N*D matrix of N data points in D dimensions.

    mu is a K*D matrix of initial cluster centers, K is
    the desired number of clusters.

    this function should return a tuple (mu, z, obj) where mu is the
    final cluster centers, z is the assignment of data points to
    clusters, and obj[i] is the kmeans objective function:
      (1/N) sum_n || x_n - mu_{z_n} ||^2
    at iteration [i].

    mu[k,:] is the mean of cluster k
    z[n] is the assignment (number in 0...K-1) of data point n

    you should run at *most* 100 iterations, but may run fewer
    if the algorithm has converged
    '''

    mu = mu0.copy()    # for safety

    N,D = X.shape
    K   = mu.shape[0]

    # initialize assignments and objective list
    z   = np.zeros((N,), dtype=int)
    obj = []

    # run at most 100 iterations
    for it in range(100):
        # store the old value of z so we can check convergence
        z_old = z.copy()
        
        # Recompute the assignment of points to centers
        old_dist = np.finfo('d').max
        for n in range(N):
            for k in range(K):
                if (old_dist > np.linalg.norm(X[n,:] - mu[k,:])):
                    old_dist = np.linalg.norm(X[n,:] - mu[k,:])
                    z[n] = k
            old_dist = np.finfo('d').max

        # Recompute means
        for k in range(K):
            temp = X[np.where(z == k)]
            for d in range(D):
                mu[k][d] = np.mean(temp[:, d], axis = 0)

        # compute the objective
        currentObjective = 0
        for n in range(N):
            currentObjective = currentObjective + np.linalg.norm(X[n,:] - mu[z[n],:]) ** 2 / float(N)
        obj.append(currentObjective)

        if verbose:
            print('Iteration %d, objective=%g' % (it, currentObjective))
        if doPlot:
            util.plotDatasetClusters(X, mu, z)
            plt.show(block=False)
            x = input("Press enter to continue...")
            if x == "q":
                doPlot = False

        # check to see if we've converged
        if all(z == z_old):
            break

    if doPlot and D==2:
        util.plotDatasetClusters(X, mu, z)
        plt.show(block=False)

    util.plotDatasetClusters(X, mu, z)
    # return the required values
    return (mu, z, np.array(obj))

def initialize_clusters(X, K, method):
    '''
    X is N*D matrix of data
    K is desired number of clusters (>=1)
    method is one of:
      determ: initialize deterministically (for comparitive reasons)
      random: just initialize randomly
      ffh   : use furthest-first heuristic

    returns a matrix K*D of initial means.

    you may assume K <= N
    '''

    N,D = X.shape
    mu = np.zeros((K,D))

    if method == 'determ':
        # just use the first K points as centers
        mu = X[:K].copy()     # be sure to copy otherwise bad things happen!!!

    elif method == 'random':
        # pick K random centers
        X_ = X.copy() # ditto above
        for k in range(1, K):
            mu[k,:] = X[int(np.random.rand() * N), :].copy()

    elif method == 'ffh':
        # pick the first center randomly and each subsequent
        # subsequent center according to the furthest first
        # heuristic
        z   = np.zeros((N,), dtype=float)
        # pick the first center totally randomly
        mu[0,:] = X[int(np.random.rand() * N), :].copy()    # be sure to copy!

        # pick each subsequent center by ldh
        for k in range(1, K):
            # find m such that data point n is the best next mean, set
            # this to mu[k,:]
            
            old_dist = np.finfo('d').max
            for n in range(N):
                for i in range(0, k):
                    if (old_dist > np.linalg.norm(X[n,:] - mu[i,:])):
                        old_dist = np.linalg.norm(X[n,:] - mu[i,:])
                z[n] = old_dist
                old_dist = np.finfo('d').max

            mu[k,:] = X[np.argmax(z), :].copy()

    elif method == 'km++':
        # pick the first center randomly and each subsequent
        # subsequent center according to the kmeans++ method
        z   = np.zeros((N,), dtype=float)
        probtable   = np.zeros((N,), dtype=float)
        # pick the first center totally randomly
        mu[0,:] = X[int(np.random.rand() * N), :].copy()    # be sure to copy!

        # pick each subsequent center by ldh
        for k in range(1, K):
            # find m such that data point n is the best next mean, set
            # this to mu[k,:]
            old_dist = np.finfo('d').max
            for n in range(N):
                for i in range(0, k):
                    if (old_dist > np.linalg.norm(X[n,:] - mu[i,:])):
                        old_dist = np.linalg.norm(X[n,:] - mu[i,:])
                z[n] = old_dist
                old_dist = np.finfo('d').max

            probtable = z/np.sum(z)
            rand_idx = np.random.multinomial(1, probtable)
            k_idx = np.where(rand_idx == 1)[0]
            mu[k,:] = X[k_idx, :].copy()
        
    else:
        print("Initialization method not implemented")
        exit(1)

    return mu


