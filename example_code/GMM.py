import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

"""
Gaussian mixture model. Currently trained using the expectation-maximisation algorithm.

P.L.Green
"""

class GMM:

    # Initialiser class method #
    def __init__(self, X, mu_init, C_init, pi_init, N_mixtures):
        self.X = np.vstack(X)   # Inputs always vertically stacked (more convenient) 
        self.mu = mu_init       # Initial means of Gaussian mixture
        self.C = C_init         # Initial covariance matrices of Gaussian mixture
        self.pi = pi_init       # Initial mixture proportions of Gaussian mixgure
        self.N_mixtures = N_mixtures            # Number of components in mixture
        self.N,self.D = np.shape(self.X)        # Number of data points and dimension of problem
        self.EZ = np.zeros([self.N,N_mixtures]) # Initialise expected labels

    # The 'E' part of the EM algorithm
    def expectation(self):
        for n in range(self.N):
            den = 0.0
            for k in range(self.N_mixtures):
                den += self.pi[k] * multivariate_normal.pdf(self.X[n], self.mu[k], self.C[k])
            for k in range(self.N_mixtures):
                num = self.pi[k] * multivariate_normal.pdf(self.X[n], self.mu[k], self.C[k])
                self.EZ[n,k] = num/den

    # The 'M' part of the EM algorithm, here we use L to represent labels 
    def maximisation(self,X,L):
        for k in range(self.N_mixtures):
            Nk = np.sum(L[:,k])
            self.pi[k] = Nk / self.N
    
            # Note - should vectorize this next bit in the future as it will be a lot faster
            self.mu[k] = 0.0
            for n in range(self.N):
                self.mu[k] += 1/Nk * L[n,k]*X[n]
            self.C[k] = np.zeros([self.N_mixtures,self.N_mixtures])
            for n in range(self.N):
                self.C[k] += 1/Nk * L[n,k]* np.vstack(X[n]-self.mu[k])*(X[n]-self.mu[k])
        
    # Train Gaussian mixture model using the EM algorithm #
    def train(self, Ni):
        print('Training...')
        for i in range(Ni):
            print('Iteration', i)
            self.expectation()
            self.maximisation(self.X,self.EZ)
         
    # Plot results (if we can) #
    def plot(self):
        if self.D == 2:
            
            # Plot contours
            r1 = np.linspace(-5,5,100)
            r2 = np.linspace(-5,5,100)
            x_r1, x_r2 = np.meshgrid(r1,r2)
            pos = np.empty(x_r1.shape + (2,))
            pos[:,:,0] = x_r1; pos[:,:,1] = x_r2
            for k in range(self.N_mixtures):
                p = multivariate_normal(self.mu[k],self.C[k])
                plt.contour(x_r1,x_r2,p.pdf(pos))

            plt.show()
            
        else:
            print('Currently only produce plots for 2D problems.')
        
        
