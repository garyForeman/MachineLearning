#! /usr/bin/env python

"""
Author: Gary Foreman
Last Modified: January 18, 2015
Solution to Exercise 9 of Andrew Ng's Machine Learning course on OpenClassroom
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os

K = 16 #number of means used for k-means algorithm
DATA_DIR = 'ex9Data/'
BIRD_SMALL = DATA_DIR + 'bird_small.tiff'
BIRD_LARGE = DATA_DIR + 'bird_large.tiff'

class K_Means(object):
    def __init__(self, image_file_name):
        self.image = plt.imread(image_file_name)
        self.k_means = np.random.randint(0, 255, (K, 3))

    def _nearest_means(self, image):
        """
        Creates array c of size image containing the index of the nearest mean
        """
        n = len(image)
        c = np.empty((n, n), np.int8)
        for i in xrange(n):
            for j in xrange(n):
                distances = np.empty(K)
                for k in xrange(K):
                    distances[k] = np.linalg.norm(image[i,j] - 
                                                  self.k_means[k])
                c[i,j] = np.argmin(distances)
        return c

    def _means_update(self, c):
        """
        Updates self.k_means by averaging the colors from self.image and their
        nearest means given in c.
        """
        for i in xrange(K):
            c_i = c == i
            n_i = np.sum(c_i)
            if n_i > 0:
                self.k_means[i] = np.sum(self.image * 
                                         np.dstack((c_i, c_i, c_i)),
                                         axis=(0,1)) / n_i

    def run(self, iter_max=100, atol=1e-7):
        """
        Runs the k-means algorithm until convergence or until iter_max is
        reached
        """
        i = 0
        epsilon = atol * 100.
        while(epsilon > atol and i < iter_max):
            k_means_old = np.copy(self.k_means)
            c = self._nearest_means(self.image)
            self._means_update(c)
            epsilon = np.linalg.norm(np.sum(self.k_means - k_means_old, axis=1))
            i += 1

    def image_compress(self, image_file_name):
        """
        Converts colors of image_file_name to colors in self.k_means. Use only
        after running K_Means.run, otherwise, solution will be nonsensical.
        """
        im_comp = plt.imread(image_file_name)
        plt.imshow(im_comp)
        plt.show()
        plt.clf()

        n = len(im_comp)
        c = self._nearest_means(im_comp)
        for i in xrange(n):
            for j in xrange(n):
                im_comp[i,j] = self.k_means[c[i,j]]

        plt.imshow(im_comp)
        plt.show()
        plt.clf()

        fn, ext = os.path.splitext(image_file_name)
        outfile = '%s_kmeans%s' % (fn, ext)
        plt.imsave(outfile, im_comp)

if __name__ == '__main__':
    k_means = K_Means(BIRD_SMALL)
    k_means.run()
    k_means.image_compress(BIRD_LARGE)
