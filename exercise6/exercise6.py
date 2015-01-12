#! /usr/bin/env python

"""
Author: Gary Foreman
Last Modified: January 12, 2015
Solution to Exercise 6 of Andrew Ng's Machine Learning course on OpenClassroom.
Note: This solution makes use of my own version of the prepared data. Simply
using the prepared data that Prof. Ng provided will crash the program because
of off-by-one errors.
"""

from __future__ import print_function
import numpy as np

NUM_TRAIN_DOCS = 700 #total
NUM_TOKENS = 2500
NUM_TEST_DOCS = 260
PREP_DIR = 'ex6DataGenerated/'
TRAIN_FEATURES = PREP_DIR + 'train-features.txt'
TRAIN_LABELS = PREP_DIR + 'train-labels.txt'
TEST_FEATURES = PREP_DIR + 'test-features.txt'
TEST_LABELS = PREP_DIR + 'test-labels.txt'

def create_train_matrix():
    train_matrix = np.zeros((NUM_TRAIN_DOCS, NUM_TOKENS), dtype=np.int32)
    for line in open(TRAIN_FEATURES):
        line = map(int, line.split())
        train_matrix[line[0], line[1]] = line[2]
    return train_matrix

def create_train_labels():
    train_labels = np.empty(NUM_TRAIN_DOCS, dtype=np.int32)
    for i, line in enumerate(open(TRAIN_LABELS)):
        line = map(int, line.split())
        train_labels[i] = line[0]
    return train_labels

def init_log_phi_given_y(train_matrix, train_labels, spam=True):
    if spam:
        condition_vector = train_labels
    else:
        condition_vector = 1 - train_labels

    phi_given_y = np.ones(NUM_TOKENS)
    denominator = np.sum(np.sum(train_matrix, axis=1) * condition_vector) + \
                  NUM_TOKENS
    for i in xrange(NUM_TOKENS):
        phi_given_y[i] += np.sum(train_matrix[:, i] * condition_vector)
    phi_given_y /= denominator

    return np.log(phi_given_y)

def init_log_phi_y(train_labels, spam=True):
    if spam:
        phi_y = np.sum(train_labels) / float(len(train_labels))
    else:
        phi_y = np.sum(1. - train_labels) / float(len(train_labels))
    return np.log(phi_y)

def classify(train_matrix, train_labels, test_features):
    log_phi_given_spam = init_log_phi_given_y(train_matrix, train_labels)
    log_phi_given_nonspam = init_log_phi_given_y(train_matrix, train_labels,
                                                 False)
    log_phi_spam = init_log_phi_y(train_labels)
    log_phi_nonspam = init_log_phi_y(train_labels, False)

    p_spam = np.zeros(NUM_TEST_DOCS)
    p_nonspam = np.zeros(NUM_TEST_DOCS)
    for file_number, word_number, word_count in test_features:
        p_spam[file_number] += word_count * log_phi_given_spam[word_number]
        p_nonspam[file_number] += word_count * \
                                  log_phi_given_nonspam[word_number]

    #print(p_spam)
    #print(p_nonspam)
    return p_spam + log_phi_spam > p_nonspam + log_phi_nonspam

if __name__ == '__main__':
    train_matrix = create_train_matrix()
    train_labels = create_train_labels()
    test_features = np.genfromtxt(TEST_FEATURES, usecols=(0,1,2), dtype=np.int)
    test_labels = np.genfromtxt(TEST_LABELS, usecols=(0), dtype=np.int)

    #print(classify(train_matrix, train_labels, test_features))
    print(np.sum(np.abs(test_labels - 
                        classify(train_matrix, train_labels, test_features))))
