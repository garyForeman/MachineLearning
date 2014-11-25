#! /usr/bin/env python

"""
Author: Gary Foreman
Last Modified: November 24, 2014
Solution to Exercise 2 of Andrew Ng's Machine Learning course on OpenClassroom
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np


AGE_FILE = 'ex2x.dat'
HEIGHT_FILE = 'ex2y.dat'
ALPHA = 0.07 #learning rate
EPSILON = np.finfo(float).eps #machine epsilon

def load_data():
    """Load data for problem"""
    x = np.genfromtxt(AGE_FILE, usecols=(0))
    y = np.genfromtxt(HEIGHT_FILE, usecols=(0))

    return x, y

def plot1(x, y):
    """Initial plot of age vs. height data"""
    plt.plot(x, y, 'o', label='Training data')
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    #plt.show()

def gradient_descent(theta, x, y):
    """Implementation of the gradient descent algorithm"""
    m = len(y)
    delta = 1. / m * np.sum((theta.T.dot(x) - y.T) * x, axis=1)
    return theta - ALPHA * delta

def plot2(x, theta):
    """Plot including the regression curve"""
    plt.plot(x[1,:], theta.T.dot(x), label='Linear regression')
    plt.legend(loc='lower right')
    plt.show()

def hypothesis(x, theta):
    """Return the estimated height given the age and the result of the
    regression."""
    return theta[0] + theta[1] * x

if __name__ == '__main__':
    age, height = load_data()
    plot1(age, height)

    m = len(height) #store the number of training examples
    age = np.array([np.ones(m), age]) #Add a column of ones to age

    theta = np.zeros(2)
    
    theta = gradient_descent(theta, age, height)
    print('After first iteration, theta =', theta)

    difference = 1.
    iteration = 0

    #absolute difference greater that machine epsilon
    while(difference > EPSILON):
        theta_old = theta
        theta = gradient_descent(theta, age, height)
        difference = np.linalg.norm(theta - theta_old)
        iteration += 1

    print('theta converges to', theta)
    plot2(age, theta)

    print('The height of a 3.5 year-old boy is %6.4f meters.' %
          hypothesis(3.5, theta))
    print('The height of a 7 year-old boy is %6.4f meters.' %
          hypothesis(7., theta))
