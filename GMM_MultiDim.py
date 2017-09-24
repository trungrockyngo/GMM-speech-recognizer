# Trung Ngo, Wed 21st, 2016

import pylab as pl
import numpy as np
import time

class GMM_MultiDim:
     # N - number of datapoints
     # D - number of dimension
    def __init__(self, N, D):
        self.N = N
        self.D = D

    def Gaussian(self, y_k, mu_k, cov_mat):
        # compute the determinant of covariance matrix and its inverse
        sigma_det = np.linalg.det(cov_mat) # turns to be cancelled in both denominator and nominator
        sigma_inv = np.linalg.inv(cov_mat)
        # print('The determinant of the covariance matrix is {0}, and its inverse is {1}'.format(sigma_det, sigma_inv))

        denom = (2* np.pi) ** (D/2) * (np.sqrt(sigma_det))
        nenom = np.exp( -0.5 * np.transpose(y_k - mu_k) * np.linalg.inv(cov_mat) * (y_k - mu_k) )
        return nenom / denom

    # K - number of groups for clustering
    def GMM_MultiDim(self, K):
        N = self.N
        D = self.D

        # initilization: datapoints = matrix of N x D
        y = 1. * np.zeros(N)
        mu = 1. * np.zeros(N)
        sigma = 1. * np.zeros(N)

        # modeling cant be formed into 2 Gaussians
        # each y[i] has its own mean mu[i] and variance sigma[i]
        for i in range(N):
            y[i] = 1. * np.zeros(D)
            # mean and variance are both D-dimension vector
            mu[i] = y[np.random.randint(0, N-1, 1)]
            sigma[i] = np.sum((y[i] - np.mean(y[i]))**2)/N
            assert mu[i].shape == (1, D)
            assert sigma[i].shape == (1, D)
            print('The mean for the vector y[{0}] is {1}, and its variance is {2}.'.format(i, mu[i], sigma[i]))

        # compute the covariance matrix
        # cov_mat = np.cov(sigma)
        # assert cov_mat.shape == (N, N)

        temp_sigma = 1. * np.zeros(K)
        cov_mat = 1. * np.zeros(K)
        # mu_cluster = 1. * np.zeros(K)

        for k in range(K):
            temp_sigma[k] = sigma[k]
            # mu_cluster[k] = mu[k]
        cov_mat = np.cov(temp_sigma)
        assert cov_mat.shape == (K,K)
        print('The covariance matrix for {0}-clustering is {1}'.format(K, cov_mat))

        # normalize the learning weight
        alpha = 1. * np.zeros(K)
        for i in range(K):
            alpha[i] = 1/K

        count = 0
        niters = 20 # fixed value of number of iterations (still plausible to be changed)
        weight = 1. * np.zeros(N)
        multi_gaussian = 1. * np.zeros(K)

        likelihood =  1. * np.zeros(niters)

        while count < niters:
            count += 1

            # generalized E-step
            for k in range(K):
                multi_gaussian[k] = 1. * np.zeros(N)

                for i in range(N):
                    multi_gaussian[k][i] = Gaussian(y[i], mu[i], cov_mat)

                    weight[i] = alpha[i] * multi_gaussian[k][i] / np.sum( alpha[i] * Gaussian(y[i], mu[i], cov_mat) )
                    print ('The shape of weight[{0}] is {1}'. format(i, weight[i].shape))
            # generalized M-step
            alpha = np.sum(weight) / N

            for k in range(K):
                mu[k] = (1/ np.sum(weight)) * np.sum( weight[k] * y )
                sigma[k] = (1/ np.sum(weight)) * (np.sum(  weight[k] * ( y[i] - mu[k] ) * np.transpose(y[i] - mu[k])  ))

            # likelihood between these K Gaussian models at the last iteration
            likelihood[count - 1 ] = np.sum( np.log( np.sum( alpha * weight ) ) )
        return likelihood
