#! /usr/bin/env python

"""
Author: Gary Foreman
Last Modified: January 12, 2015
Prepares feature vectors from raw email files packaged in ex6DataEmails.zip.
Writes features to files similar to those found in ex6DataPrepared.zip.
"""

from __future__ import print_function
from os import listdir
from os.path import isfile, join


DATA_DIR = 'ex6DataEmails/'
OUT_DIR = 'ex6DataGenerated/'
EMAIL_DIRS = ['nonspam-train/', 'spam-train/', 'nonspam-test/', 'spam-test/']
DICT_LENGTH = 2500
TRAIN_FILES = 350 #of each type
TEST_FILES = 130
TEST_FEATURES = 'test-features.txt'
TRAIN_FEATURES = 'train-features.txt'
TEST_LABELS = 'test-labels.txt'
TRAIN_LABELS = 'train-labels.txt'

def load_email(path):
    """Opens email located at path, and returns conents as a string."""
    with open(path) as infile:
        content = infile.read()
    return content

def generate_dictionary(content, dict_name):
    """
    Adds words in string content to dict_name. The keys are the words, and
    the values are the number of times each word appears
    """
    content = content.split()
    for word in content:
        if word in dict_name:
            dict_name[word] += 1
        else:
            dict_name[word] = 1

def file_generator(email_dirs, train=False):
    """Yields all files in the directories of the list email_dirs"""
    for e_dir in email_dirs:
        directory = DATA_DIR + e_dir
        files = [directory + f for f in listdir(directory)
                 if isfile(join(directory, f))]
        email_number = 0
        for file_name in files:
            if train and email_number < TRAIN_FILES:
                yield file_name
                email_number += 1
            elif not train:
                yield file_name

def full_dictionary():
    """
    Returns a dictionary containing the 2500 most frequent words found in all
    860 email examples. The keys are the words, and the values are each words
    rank in terms of frequency.
    """
    full_dictionary = {} #contains all words in all emails
    return_dictionary = {} #contains DICT_LENGTH most frequent words
    for email in file_generator(EMAIL_DIRS):
        content = load_email(email)
        generate_dictionary(content, full_dictionary)

    for key in full_dictionary.keys():
        if len(key) == 1:
            del(full_dictionary[key])

    top_keys = sorted(full_dictionary, key=full_dictionary.get,
                      reverse=True)[:2500]
    for i, key in enumerate(top_keys):
        return_dictionary[key] = i

    return return_dictionary

def label_files():
    """
    Function to create the *-labels.txt files. The line number corresponds to
    the file id, a value of 0 means the email is not spam, and a value of 1
    means the email is spam.
    """
    with open(OUT_DIR + TEST_LABELS, 'w') as outfile:
        for i in xrange(TEST_FILES):
            outfile.write('0\n')
        for i in xrange(TEST_FILES):
            outfile.write('1\n')

    with open(OUT_DIR + TRAIN_LABELS, 'w') as outfile:
        for i in xrange(TRAIN_FILES):
            outfile.write('0\n')
        for i in xrange(TRAIN_FILES):
            outfile.write('1\n')

def feature_files():
    """
    Creates the *-features.txt files.
    Column 1: file identifier
    Column 2: word identifier
    Column 3: word count
    """
    filter_dictionary = full_dictionary()
    file_list = [OUT_DIR + TRAIN_FEATURES, OUT_DIR + TEST_FEATURES]
    for j, file_name in enumerate(file_list):
        train = j == 0
        with open(file_name, 'w') as outfile:
            for i, email in enumerate(file_generator(EMAIL_DIRS[j*2:(j+1)*2],
                                                     train)):
                content = load_email(email)
                email_dict = {}
                generate_dictionary(content, email_dict)
                for key in email_dict.keys():
                    if key in filter_dictionary.keys():
                        outfile.write('%d %d %d\n' % 
                                      (i, filter_dictionary[key],
                                       email_dict[key]))

if __name__ == '__main__':
    label_files()
    feature_files()
