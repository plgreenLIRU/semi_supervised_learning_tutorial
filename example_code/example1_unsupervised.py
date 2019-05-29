import numpy as np
from matplotlib import pyplot as plt
from numpy.random import multivariate_normal as mvn
from GMM import *
import pickle

"""
Example 1. Standard unsupervised Gaussian Mixture Model. 

P.L.Green
"""

# Make some 2D data from a mixture of 2 Gaussians.
mu1 = np.array([2, 2])
mu2 = np.array([-3, -2])
C1 = np.array([[1, -0.7], [-0.7, 1]])
C2 = np.eye(2)
pi1 = 0.7
pi2 = 0.3
N = 300
X = np.zeros([N, 2])
for i in range(N):
    u = np.random.rand()
    if u < pi1:
        X[i] = mvn(mu1, C1)
    else:
        X[i] = mvn(mu2, C2)
        
# Create and train GMM object
mu = [np.array([1, 1]), np.array([-1, -1])]
C = [np.eye(2), np.eye(2)]
pi = np.array([0.5, 0.5])
gmm = GMM(X=X, mu_init=mu, C_init=C, pi_init=pi, N_mixtures=2)
gmm.train(Ni=5)

# Print and plot results
for k in range(2):
    print('\nMean', k+1, ' = ', gmm.mu[k], '\n')
    print('Covariance matrix', k+1, ' = \n', gmm.C[k], '\n')
    print('Mixture proportion', k+1, ' = ', gmm.pi[k], '\n')
gmm.plot()
plt.show()
