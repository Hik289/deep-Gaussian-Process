from __future__ import print_function

import numpy as np
from scipy import array, linalg, dot


def covariance_function(x1, x2, log_theta_sigma2, log_theta_lengthscale):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    K = np.zeros([n1, n2])
    for i in range(n1):
        for j in range(n2):
            K[i, j] = np.exp(log_theta_sigma2) * np.exp(- 0.5 * np.dot(x1[i,:] - x2[j,:], x1[i,:] - x2[j,:]) / np.exp(2.0 * log_theta_lengthscale))
    return K
def log_p_Y_given_F1(Y, F1, log_theta):
    log_theta_sigma2, log_theta_lengthscale, log_theta_lambda = unpack_log_theta(log_theta)
    n = Y.shape[0]
    K_Y = covariance_function(F1, F1, log_theta_sigma2[1], log_theta_lengthscale[1]) + np.eye(n) * np.exp(log_theta_lambda)
    L_K_Y, lower_K_Y = linalg.cho_factor(K_Y, lower=True)
    nu = linalg.solve_triangular(L_K_Y, Y, lower=True)
    return -np.sum(np.log(np.diagonal(L_K_Y))) - 0.5 * np.dot(nu.transpose(), nu)
def do_sampleF1(Y, X, current_F1, log_theta):
    log_theta_sigma2, log_theta_lengthscale, log_theta_lambda = unpack_log_theta(log_theta)    
    n = Y.shape[0]
    current_logy = log_p_Y_given_F1(Y, current_F1, log_theta) + np.log(np.random.uniform(0.0, 1.0, 1))
    proposed_F1 = current_F1 * 1.0
    K_F1 = covariance_function(X, X, log_theta_sigma2[0], log_theta_lengthscale[0]) + np.eye(n) * 1e-9
    L_K_F1 = linalg.cholesky(K_F1, lower=True)
    auxiliary_nu = np.dot(L_K_F1, np.random.normal(0.0, 1.0, [n,1]))
    auxiliary_theta = np.random.uniform(0.0, 2 * np.pi, 1)
    auxiliary_thetamin = auxiliary_theta - 2 * np.pi
    auxiliary_thetamax = auxiliary_theta * 1.0
    while True:
        proposed_F1 = current_F1 * np.cos(auxiliary_theta) + auxiliary_nu * np.sin(auxiliary_theta)
        proposed_logy = log_p_Y_given_F1(Y, proposed_F1, log_theta)
        if proposed_logy > current_logy:
            break    
        if(auxiliary_theta < 0):
            auxiliary_thetamin = auxiliary_theta * 1.0
        if(auxiliary_theta >= 0):
            auxiliary_thetamax = auxiliary_theta * 1.0        
        auxiliary_theta = np.random.uniform(auxiliary_thetamin, auxiliary_thetamax, 1)
    return proposed_F1
def do_sampleF2(Y, X, current_F1, log_theta):
    log_theta_sigma2, log_theta_lengthscale, log_theta_lambda = unpack_log_theta(log_theta)
    n = Y.shape[0]
    K_F2 = covariance_function(current_F1, current_F1, log_theta_sigma2[1], log_theta_lengthscale[1]) + np.eye(n) * 1e-9
    K_Y = K_F2 + np.eye(n) * np.exp(log_theta_lambda)
    L_K_Y, lower_K_Y = linalg.cho_factor(K_Y, lower=True)
    K_inv_Y = linalg.cho_solve((L_K_Y, lower_K_Y), Y)
    mu = np.dot(K_F2, K_inv_Y)
    K_inv_K = linalg.cho_solve((L_K_Y, lower_K_Y), K_F2)
    Sigma = K_F2 - np.dot(K_F2, K_inv_K)
    L_Sigma = linalg.cholesky(Sigma, lower=True)
    proposed_F2 = mu + np.dot(L_Sigma, np.random.normal(0.0, 1.0, [n, 1]))
    return proposed_F2
def unpack_log_theta(log_theta):
    return log_theta[0], log_theta[1], log_theta[2]
def MCMC(X, Y, Xtest, n_MCMC = 100, nburnin = 10, save_every = 10):
    n = X.shape[0]
    ntest = Xtest.shape[0]
    log_theta_sigma2 = np.loadtxt("./mcmc/log_theta_sigma2.txt", delimiter='\t')
    log_theta_lengthscale = np.loadtxt("./mcmc/log_theta_lengthscale.txt", delimiter='\t')
    log_theta_lambda = np.loadtxt("./mcmc/log_lambda.txt", delimiter='\t')    
    log_theta = (log_theta_sigma2, log_theta_lengthscale, log_theta_lambda)
    samples_F1 = np.zeros([n, n_MCMC])
    samples_F2 = np.zeros([n, n_MCMC])
    predictions_F1 = np.zeros([ntest, n_MCMC])
    predictions_F2 = np.zeros([ntest, n_MCMC])
    current_F1 = np.zeros([n, 1])
    current_F2 = np.zeros([n, 1])
    for iteration_MCMC in range(-nburnin,n_MCMC):       
        for inner_iteration_MCMC in range(save_every):
            current_F1 = do_sampleF1(Y, X, current_F1, log_theta)
            current_F2 = do_sampleF2(Y, X, current_F1, log_theta)
        if iteration_MCMC >= 0:
            samples_F1[:,iteration_MCMC] = current_F1[:,0]
            samples_F2[:,iteration_MCMC] = current_F2[:,0]
            K_F1 = covariance_function(X, X, log_theta_sigma2[0], log_theta_lengthscale[0]) + np.eye(n) * 1e-9
            L_F1, lower_K_F1 = linalg.cho_factor(K_F1, lower=True)
            K_star = covariance_function(X, Xtest, log_theta_sigma2[0], log_theta_lengthscale[0])
            K_star_star = covariance_function(Xtest, Xtest, log_theta_sigma2[0], log_theta_lengthscale[0]) + np.eye(ntest) * 1e-9
            L_inv_K_star = linalg.solve_triangular(L_F1, K_star, lower=True)
            L_inv_F1 = linalg.solve_triangular(L_F1, current_F1, lower=True)
            mu_star = np.dot(L_inv_K_star.transpose(), L_inv_F1)
            Sigma_star = K_star_star - np.dot(L_inv_K_star.transpose(), L_inv_K_star)
            L_Sigma_star = linalg.cholesky(Sigma_star, lower=True)
            predictions_F1[:,iteration_MCMC] = (np.dot(L_Sigma_star, np.random.normal(0, 1, [ntest, 1])) + mu_star)[:,0]
            K_F2 = covariance_function(current_F1, current_F1, log_theta_sigma2[1], log_theta_lengthscale[1]) + np.eye(n) * 1e-9
            L_K_F2, lower_K_F2 = linalg.cho_factor(K_F2, lower=True)
            TMP = predictions_F1[:,iteration_MCMC]
            TMP = np.reshape(TMP, [ntest,1])
            K_star = covariance_function(current_F1, TMP, log_theta_sigma2[1], log_theta_lengthscale[1])
            K_star_star = covariance_function(TMP, TMP, log_theta_sigma2[1], log_theta_lengthscale[1]) + np.eye(ntest) * 1e-9
            L_inv_K_star = linalg.solve_triangular(L_K_F2, K_star, lower=True)
            L_inv_F2 = linalg.solve_triangular(L_K_F2, current_F2, lower=True)
            mu_star = np.dot(L_inv_K_star.transpose(), L_inv_F2)
            Sigma_star = K_star_star - np.dot(L_inv_K_star.transpose(), L_inv_K_star)
            L_Sigma_star = linalg.cholesky(Sigma_star, lower=True)
            predictions_F2[:,iteration_MCMC] = (np.dot(L_Sigma_star, np.random.normal(0, 1, [ntest, 1])) + mu_star)[:,0]
    return samples_F1, samples_F2, predictions_F1, predictions_F2
