#! /usr/bin/env python

"""
Author: Gary Foreman
Last Modified: January 16, 2015
Solution to Exercise 8 of Andrew Ng's Machine Learning course on OpenClassroom
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

DATA_DIR = 'ex8Data/'
PART_A_FILE = DATA_DIR + 'ex8a.txt'
PART_B_FILE = DATA_DIR + 'ex8b.txt'

def load_data(file_name, attribute_number):
    """
    Reads data from file_name file. Note: files are formatted for use
    with the LIBSVM Matlab interface. This is why the second and third columns
    are read starting with their third elements.
    """
    train_labels = np.genfromtxt(file_name, usecols=(0), dtype=np.int)
    train_features = np.zeros((len(train_labels), attribute_number))
    for i, line in enumerate(open(file_name)):
        line = line.split()
        for j in xrange(1, len(line)):
            index, value = map(float, line[j].split(':'))
            train_features[i, int(index)-1] = value

    return train_labels, train_features

def plot_data(train_labels, train_features):
    """Generates 2-D scatter plot of train features colored by train_labels"""
    pos = train_labels == 1
    neg = train_labels == -1
    plt.plot(train_features[pos, 0], train_features[pos, 1], 'ro')
    plt.plot(train_features[neg, 0], train_features[neg, 1], 'go')

def plot_boundary(labels, features, model, varargin=False):
    """
    Generates plot of the decision boundary on top of the scatter plot from
    plot_data. Function adapted from octave script, plotboundary.m, supplied
    by Andrew Ng 
    """
    x_plot = np.linspace(np.min(features[:,0]), np.max(features[:,0]), 100)
    y_plot = np.linspace(np.min(features[:,1]), np.max(features[:,1]), 100)
    X, Y = np.meshgrid(x_plot, y_plot)
    vals = np.zeros((100, 100))
    for i in xrange(np.size(X, axis=1)):
        x = np.reshape(np.append(X[:,i], Y[:,i]), (2, 100)).T
        vals[:,i] = model.decision_function(x).flatten()

    if varargin:
        plt.contourf(X, Y, vals, 50, ls='None', cmap='bone')
    plt.contour(X, Y, vals, [0], lw=2, colors='black')
    plot_data(labels, features)
    plt.title('$\gamma$ =  %d' % model.gamma, size='x-large')
    plt.show()
    plt.clf()

if __name__ == '__main__':
    #Part A
    train_labels, train_features = load_data(PART_A_FILE, 2)
    gamma = 100.
    model = svm.SVC(gamma=gamma) #default kernel is rbf
    model.fit(train_features, train_labels)
    plot_boundary(train_labels, train_features, model, True)

    #Part B
    train_labels, train_features = load_data(PART_B_FILE, 2)
    gamma_list = [1, 10, 100, 1000]
    for gamma in gamma_list:
        model = svm.SVC(gamma=gamma) #default kernal is rbf
        model.fit(train_features, train_labels)
        plot_boundary(train_labels, train_features, model)
