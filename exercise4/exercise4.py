#! /usr/bin/env python

"""
Author: Gary Foreman
Last Modified: December 7, 2014
Solution to Exercise 4 of Andrew Ng's Machine Learning course on OpenClassroom
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

X_FILE = 'ex4x.dat'
Y_FILE = 'ex4y.dat'

def load_data():
    """Load data for problem"""
    x = np.genfromtxt(X_FILE, usecols=(0, 1))
    y = np.genfromtxt(Y_FILE, usecols=(0))

    return x, y

def plot_data(x, pos, neg):
    """Generate plot of raw data"""
    plt.plot(x[pos, 1], x[pos, 2], '+', label='Admitted')
    plt.plot(x[neg, 1], x[neg, 2], 'bo', label='Not admitted')
    plt.xlim(xmin=15, xmax=65)
    plt.ylim(ymin=40, ymax=100)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    #plt.legend(loc='upper left')
    #plt.show()
    #plt.clf()

def g(z):
    """Hypothesis function for logistic regression"""
    return 1. / (1. + np.exp(-z))

def J(theta, x, y):
    """Cost function for logistic regression"""
    m = len(y)
    z = theta.dot(x.T) #argument for hypothesis function
    return 1. / m * np.sum(-y * np.log(g(z)) - (1. - y) * np.log(1 - g(z)))

def gradient(theta, x, y):
    """Gradient of the cost function"""
    m = len(y)
    n = len(theta)
    z = theta.dot(x.T)
    grad = np.zeros(n)
    for i in xrange(m):
        grad += (g(z[i]) - y[i]) * x[i]
    return 1. / m * grad

def Hessian(theta, x, y):
    m = len(y)
    n = len(theta)
    z = theta.dot(x.T)
    H = np.zeros((n, n))
    for i in xrange(m):
        H += g(z[i]) * (1. - g(z[i])) * np.outer(x[i], x[i])
    return 1. / m * H

def plot_convergence(J_list):
    plt.plot(range(len(J_list)), J_list)
    plt.ylabel('J')
    plt.xlabel('Iteration')
    plt.show()
    plt.clf()

def decision_boundary(x, theta):
    """Returns theta.T.dot(x) = 0"""
    return -(theta[1] * x + theta[0]) / theta[2]

def plot_decision(theta, x, pos, neg):
    plot_data(x, pos, neg)
    x_dec = np.array([np.min(x[:,1]), np.max(x[:,1])])
    y_dec = decision_boundary(x_dec, theta)
    plt.plot(x_dec, y_dec, 'b-', label='Decision boundary')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    scores, admittance = load_data()
    scores = np.append(np.ones((len(scores), 1)), scores, axis=1)
    pos = admittance == 1
    neg = admittance == 0

    #plot_data(scores, pos, neg)

    theta = np.zeros(3)
    J_list = []
    for i in xrange(15):
        J_list.append(J(theta, scores, admittance))
        theta -= (np.linalg.inv(Hessian(theta, scores, admittance))
                  .dot(gradient(theta, scores, admittance)))

    plot_convergence(J_list)
    plot_decision(theta, scores, pos, neg)

    print('theta =', theta)
    print('The probability of admission for a student with Exam 1 score 20 and '
          'Exam 2 score 80 is %.4f.' % g(theta.dot(np.array([1., 20., 80.]))))
