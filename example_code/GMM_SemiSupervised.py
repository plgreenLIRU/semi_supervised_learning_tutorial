import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from GMM import *

"""
Gaussian mixture model. Currently trained using the
expectation-maximisation algorithm.

P.L.Green

"""


class GMM_SemiSupervised(GMM):

    def __init__(self, X, X_labelled, Y, mu_init, C_init, pi_init,
                 N_mixtures):
        """ Initialiser class method

        """


        self.X = np.vstack(X)
        self.X_labelled = np.vstack(X_labelled)
        self.X_all = np.vstack((self.X_labelled, self.X))
        self.Y = np.vstack(Y)
        self.mu = mu_init
        self.C = C_init
        self.pi = pi_init
        self.N_mixtures = N_mixtures
        self.N_labelled, self.D = np.shape(self.X_labelled)
        self.N = np.shape(X)[0]
        self.EZ = np.zeros([self.N, N_mixtures]) # Initialise expected labels

    def train(self, Ni):
        """ Train (using EM)

        """

        print('Training...')
        for i in range(Ni):
            print('Iteration', i)
            self.expectation()
            L = np.vstack((self.Y, self.EZ))
            self.maximisation(self.X_all, L)

    def plot(self):
        """ Plots (just for 2D where no. of mixtures is 2 for now)

        """

        super().plot()
        if self.D is 2 and self.N_mixtures is 2:
            for i in range(self.N_labelled):
                if self.Y[i, 0] == 1:
                    plt.plot(self.X_labelled[i, 0], self.X_labelled[i, 1], 'v',
                             markerfacecolor=[0, 1, 0],
                             markeredgecolor='black',
                             markersize=10)
                else:
                    plt.plot(self.X_labelled[i, 0], self.X_labelled[i, 1], 'v',
                             markerfacecolor=[0, 0, 1],
                             markeredgecolor='black',
                             markersize=10)
