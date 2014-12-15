#! /usr/bin/env python

"""
Author: Gary Foreman
Last Modified: December 14, 2014
Solution to Exercise 4 of Andrew Ng's Machine Learning course on OpenClassroom
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

LIN_X_FILE = 'ex5Linx.dat'
LIN_Y_FILE = 'ex5Liny.dat'
LOG_X_FILE = 'ex5Logx.dat'
LOG_Y_FILE = 'ex5Logy.dat'

LINEAR_POLYNOMIAL_ORDER = 5

def load_lin_data():
    """Load data for regularized linear regression"""
    x = np.genfromtxt(LIN_X_FILE, usecols=(0))
    y = np.genfromtxt(LIN_Y_FILE, usecols=(0))

    return x, y

def plot_lin_data(x, y):
    """Plot data for regularized linear regression"""
    plt.plot(x, y, 'ro', label='Training data')
    #plt.show()
    #plt.clf()

def linear_features(x):
    features = np.ones(LINEAR_POLYNOMIAL_ORDER + 1)
    for i in xrange(1, LINEAR_POLYNOMIAL_ORDER + 1):
        features[i] = features[i-1] * x
    return features

def linear_hypothesis(theta, x):
    features = linear_features(x)
    return theta.dot(features)

def linear_normal_eqs(x, y, reg_param):
    m = len(x)
    X = np.empty((m, LINEAR_POLYNOMIAL_ORDER + 1))
    for i in xrange(m):
        X[i] = linear_features(x[i])

    reg_matrix = np.identity(LINEAR_POLYNOMIAL_ORDER + 1)
    reg_matrix[0,0] = 0.
    return np.linalg.inv(X.T.dot(X) + reg_param * reg_matrix).dot(X.T).dot(y)

def linear_main():
    """Function to carry out linear regression portion of the exercise"""
    print('Regularized linear regression:')
    x, y = load_lin_data()
    plot_lin_data(x, y)
    reg_param_list = [0., 1., 10.]
    plot_x = np.linspace(-1, 1, 1000)
    for reg_param in reg_param_list:
        theta = linear_normal_eqs(x, y, reg_param)
        print(reg_param, theta)
        plot_y = [linear_hypothesis(theta, plot_x[i])
                  for i in xrange(len(plot_x))]
        plt.plot(plot_x, plot_y,
                 label=r'5th order fit, $\lambda$ = %d' % reg_param)
    plt.legend()
    plt.show()
    plt.clf()
    print()

def load_log_data():
    """Load data for regularized logistic regression"""
    x = np.genfromtxt(LOG_X_FILE, delimiter=',', usecols=(0,1))
    m = len(x)
    #x = np.append(np.ones((m, 1)), x, axis=1)
    y = np.genfromtxt(LOG_Y_FILE)
    return x, y

def plot_log_data(x, y, pos, neg):
    plt.plot(x[pos, 0], x[pos, 1], 'k+', label='y = 1')
    plt.plot(x[neg, 0], x[neg, 1], 'yo', label='y = 0')
    plt.legend()
    #plt.show()

def map_feature(feat1, feat2):
    """Maps the two input feartures to higher-order featurs as defined in
    Exercise 5. Returns a new feature array with more features. Adapted from
    map_feature.m supplied by Andrew Ng."""
    degree = 6
    try:
        out = np.ones((28, len(feat1)))
    except TypeError:
        out = np.ones(28)
    k = 1
    for i in xrange(1, degree+1):
        for j in xrange(0, i+1):
            out[k] = feat1**(i-j) * feat2**j
            k += 1
    return out

def g(z):
    """Hypothesis function for logistic regression"""
    return 1. / (1. + np.exp(-z))

def regularized_log_cost(theta, x, y, reg_param):
    """Cost function for regularized logistic regression."""
    m = len(y)
    z = theta.dot(x.T) #argument for hypothesis function
    return (-1. / m * np.sum(y * np.log(g(z)) + (1. - y) * np.log(1 - g(z))) +
            0.5 * float(reg_param) / m * np.sum(theta[1:]**2))

def logistic_gradient(theta, x, y, reg_param):
    m = len(y)
    n = len(theta)
    z = theta.dot(x.T)
    grad = np.zeros(n)
    for i in xrange(1, m):
        grad += (g(z[i]) - y[i]) * x[i]
    grad[1:] += reg_param * theta[1:]
    return grad / m

def Hessian(theta, x, y, reg_param):
    m = len(y)
    n = len(theta)
    z = theta.dot(x.T)
    H = np.zeros((n,n))
    for i in xrange(m):
        H += g(z[i]) * (1. - g(z[i])) * np.outer(x[i], x[i])
    reg_matrix = np.identity(n)
    reg_matrix[0,0] = 0.
    return (H + reg_param * reg_matrix) / m

def plot_contour(theta):
    u = np.linspace(-1, 1.5, 200)
    v = np.linspace(-1, 1.5, 200)
    z = np.zeros((len(u), len(v)))

    for i in xrange(len(u)):
        for j in xrange(len(v)):
            z[j, i] = map_feature(u[i], v[j]).dot(theta)

    plt.contour(u, v, z, levels=[0])
    plt.show()
    plt.clf()

def logistic_main():
    """Function to carry out logistic regression portion of the exercise"""
    print('Regularized logistic regression:')
    uv, y = load_log_data()
    pos = y == 1
    neg = y == 0
    #plot_log_data(uv, y, pos, neg)
    x = map_feature(uv[:,0], uv[:,1])
    x = x.T
    reg_param_list = [0., 1., 10.]
    for reg_param in reg_param_list:
        theta = np.zeros(28)
        J_list = np.zeros(15)
        for i in xrange(15):
            J_list[i] = regularized_log_cost(theta, x, y, reg_param)
            theta -= (np.linalg.inv(Hessian(theta, x, y, reg_param))
                      .dot(logistic_gradient(theta, x, y, reg_param)))
        print('lambda = %d, norm(theta) = %f' %
              (reg_param, np.linalg.norm(theta)))
        plot_log_data(uv, y, pos, neg)
        plt.title(r'$\lambda$ = %d' % reg_param)
        plot_contour(theta)

if __name__ == '__main__':
    linear_main()
    logistic_main()
