#! /usr/bin/env python

"""
Author: Gary Foreman
Last Modified: December 2, 2014
Solution to Exercise 3 of Andrew Ng's Machine Learning course on OpenClassroom
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np


M = 47 #number of training examples
X_FILE = 'ex3x.dat'
Y_FILE = 'ex3y.dat'

def load_data():
    """Load data for problem"""
    x = np.genfromtxt(X_FILE, usecols=(0, 1))
    y = np.genfromtxt(Y_FILE, usecols=(0))

    return x, y

def scale_features(x):
    """Function for scaling features so that they all lie within approximately
    the same range."""
    mean = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    for i in xrange(1, len(x[0])):
        x[:,i] = (x[:,i] - mean[i]) / sigma[i]

    return mean, sigma

def gradient_descent(theta, x, y, alpha):
    """Implementation of the gradient descent algorithm"""
    delta = 1. / M * np.sum((theta.dot(x) - y) * x, axis=1)
    return theta - alpha * delta

def cost_function(theta, x, y):
    return (0.5 / M * (x.T.dot(theta) - y)
            .dot(x.T.dot(theta) - y))

def plot_cost(x, y):
    alpha_list = np.logspace(-3, 0, 7)

    alpha_list = np.append(alpha_list, 1.3)

    for alpha in alpha_list:
        theta = np.zeros(3)
        J_values = np.zeros(50)
        for i in xrange(50):
            J_values[i] = cost_function(theta, x, y)
            theta = gradient_descent(theta, x, y, alpha)

        plt.plot(range(50), J_values, label=(r'$\alpha$ = %5.3f' % alpha))

    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.legend()
    plt.show()

def normal_equations(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y) 

def hypothesis(x, theta):
    """Return the estimated housing price given the living area, number of 
    bedrooms, and the result of the regression."""
    return x.dot(theta)

if __name__ == '__main__':
    #x[1] is the living area of homes
    #x[2] is the number of bedrooms
    #y is the sale price
    x, y = load_data()
    x = np.append(np.ones((M,1)), x, axis=1)
    mu, sigma = scale_features(x)
    x = x.T

    plot_cost(x, y)

    alpha = 1.0 #chosen based on convergence rate
    theta = np.zeros(3)
    for i in xrange(100):
        theta = gradient_descent(theta, x, y, alpha)

    print('Gradient descent:')
    print('Theta =', theta)
    example_data = np.array([1., (1650 - mu[1]) / sigma[1],
                             (3 - mu[2]) / sigma[2]])
    print('A 1650 square-foot, 3 bedroom house should cost $%.2f.' %
          hypothesis(example_data, theta))

    print('\nNormal Equaitons:')
    x, y = load_data()
    x = np.append(np.ones((M,1)), x, axis=1)
    theta = normal_equations(x,y)
    print('Theta =', theta)
    print('A 1650 square-foot, 3 bedroom house should cost $%.2f.' %
          hypothesis(np.array([1., 1650, 3]), theta))
