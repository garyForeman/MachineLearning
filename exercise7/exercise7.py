#! /usr/bin/env python

"""
Author: Gary Foreman
Last Modified: January 15, 2015
Solution to Exercise 7 of Andrew Ng's Machine Learning course on OpenClassroom
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

DATA_DIR = 'ex7Data/'
TWO_FEATURE_FILE = DATA_DIR + 'twofeature.txt'
TWO_FEATURE_ATTR_NUM = 2
EMAIL_NUMBER = ['50', '100', '400', 'all']
EMAIL_TRAIN_FILES = [DATA_DIR + 'email_train-' + i + '.txt' 
                     for i in EMAIL_NUMBER]
EMAIL_TEST_FILE = DATA_DIR + 'email_test.txt'
WORD_NUMBER = 2500

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

def plot_two_features(train_labels, train_features):
    """Generates 2-D scatter plot of train features colored by train_labels"""
    pos = train_labels == 1
    neg = train_labels == -1
    plt.plot(train_features[pos, 0], train_features[pos, 1], 'bo')
    plt.plot(train_features[neg, 0], train_features[neg, 1], 'go')

def support_vector_machine(train_labels, train_features, cost_factor):
    """
    Wrapper for sklearn.svm.SVC. Trains support vector machine using
    train_labels and train_features, using cost_factor. Generates plot of
    model on top of scatter plot from plot_two_features.
    """
    model = svm.SVC(C=cost_factor, kernel='linear')
    model.fit(train_features, train_labels)

    w = model.coef_[0]
    b = -w[0] / w[1]
    boundary_x = np.linspace(np.min(train_features[:, 0]),
                             np.max(train_features[:, 0]))
    boundary_y = b * boundary_x - model.intercept_[0] / w[1]

    plot_two_features(train_labels, train_features)
    plt.plot(boundary_x, boundary_y, 'k-')
    plt.title('SVM Linear Classifier with C = ' + str(cost_factor),
              size='x-large')
    plt.show()
    plt.clf()

def spam_filter():
    true_answers, test_features = load_data(EMAIL_TEST_FILE, WORD_NUMBER)

    for i, file_name in enumerate(EMAIL_TRAIN_FILES):
        spam, word_counts = load_data(file_name, WORD_NUMBER)
        model = svm.SVC(kernel='linear')
        model.fit(word_counts, spam)
        predicted_answers = model.predict(test_features)
        misclassified = np.abs(predicted_answers - true_answers) / 2
        print('The ' + EMAIL_NUMBER[i] + ' email training set misclassifies '+
              str(np.sum(misclassified)) + ' out of 260 emails.')

if __name__ == '__main__':
    train_labels, train_features = load_data(TWO_FEATURE_FILE,
                                             TWO_FEATURE_ATTR_NUM)
    support_vector_machine(train_labels, train_features, 1)
    support_vector_machine(train_labels, train_features, 100)

    spam_filter()
