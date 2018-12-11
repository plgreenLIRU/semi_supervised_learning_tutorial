import numpy as np
from matplotlib import pyplot as plt
from numpy.random import multivariate_normal as mvn
from GMM_SemiSupervised import *

"""
Example 2 - semi-supervised learning using a Gaussian Mixture Model

P.L.Green
"""

### Make some 2D data from a mixture of Gaussians, some labelled and some not. ###
mu1 = np.array([2,2])
mu2 = np.array([-2,-3])
C1 = np.array([ [1,-0.7],[-0.7,1] ])
C2 = np.eye(2)
pi1 = 0.7
pi2 = 0.3

N_unlabelled = 280
N_labelled = 20

X_labelled = np.zeros([N_labelled,2])
Y = np.zeros([N_labelled,2])
X_unlabelled = np.zeros([N_unlabelled,2])
for i in range(N_labelled):
    u = np.random.rand()
    if u < pi1:
        X_labelled[i] = mvn(mu1,C1)
        Y[i,0] = 1
    else:
        X_labelled[i] = mvn(mu2,C2)
        Y[i,1] = 1
for i in range(N_unlabelled):
    u = np.random.rand()
    if u < pi1:
        X_unlabelled[i] = mvn(mu1,C1)
    else:
        X_unlabelled[i] = mvn(mu2,C2)

### Initial estimates for Gaussian Mixture Model ###
mu = [ np.array([1,1]), np.array([-1,-1]) ]
C = [np.eye(2),np.eye(2)]
pi = np.array([0.5,0.5])

### Create and train Gaussian Mixture Model ###
gmm_ss = GMM_SemiSupervised(X=X_unlabelled, X_labelled=X_labelled, Y=Y, mu_init=mu, C_init=C, pi_init=pi, N_mixtures=2)
gmm_ss.train(Ni=5)

### Final plot ###
plt.plot(X_unlabelled[:,0],X_unlabelled[:,1],'o',color='black')
for i in range(N_labelled):
    if Y[i,0] == 1:
        plt.plot(X_labelled[i,0],X_labelled[i,1],'o',color='red')
    else:
        plt.plot(X_labelled[i,0],X_labelled[i,1],'o',color='blue')
gmm_ss.plot()

















