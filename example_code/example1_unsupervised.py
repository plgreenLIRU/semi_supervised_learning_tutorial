import numpy as np
from matplotlib import pyplot as plt
from numpy.random import multivariate_normal as mvn
from GMM import *

"""
Example 1 - unsupervised learning using a Gaussian Mixture Model

P.L.Green
"""

### Make some 2D data from a mixture of Gaussians ###
mu1 = np.array([2,2])
mu2 = np.array([-2,-3])
C1 = np.array([ [1,-0.7],[-0.7,1] ])
C2 = np.eye(2)
pi1 = 0.7
pi2 = 0.3

N = 500
X = np.zeros([N,2])
for i in range(N):
    u = np.random.rand()
    if u < pi1:
        X[i] = mvn(mu1,C1)
    else:
        X[i] = mvn(mu2,C2)

### Initial estimates for Gaussian Mixture Model ###
mu = [ np.array([1,1]), np.array([-1,-1]) ]
C = [np.eye(2),np.eye(2)]
pi = np.array([0.5,0.5])

### Create and train Gaussian Mixture Model ###
gmm = GMM(X=X, mu_init=mu, C_init=C, pi_init=pi, N_mixtures=2)
gmm.train(Ni=5)
plt.plot(X[:,0],X[:,1],'o')
gmm.plot()

### Print results ###
for k in range(2):
    print('\nMean', k+1, ' = ', gmm.mu[k], '\n')
    print('Covariance matrix', k+1, ' = ', gmm.C[k], '\n')
    print('Mixture proportion', k+1, ' = ', gmm.pi[k], '\n')


