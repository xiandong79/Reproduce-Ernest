#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy.optimize import nnls
import scipy
import numpy as np
import sys
import matplotlib.pyplot as plt


class train_model_and_estimate(object):
    def __init__(self, file_address):
        print 'The script is reading the data... '
        cores, fractions, runtimes = np.loadtxt(file_address, dtype=float, delimiter=',', usecols=(0, 1, 2), unpack=True, skiprows=1)
        self.cores = cores
        self.fractions = fractions
        self.runtimes = runtimes
        self.model = self.train()

    def train(self):
        train_points = np.vstack((self.cores, self.fractions)).T
        # maybe np.hstack((a,b))
        train_features = np.array([self.get_features([row[0], row[1]]) for row in train_points])
        model = nnls(train_features, self.runtimes)
        print 'The script is training the model... '
        return model[0]

    def estimate(self, test_points):
        test_features = np.array([self.get_features([row[0], row[1]]) for row in test_points])
        print 'The script is predicting the runtime of test_points... '
        return test_features.dot(self.model)


    def get_features(self, data_points):
        data_cores = data_points[0]
        data_scale = data_points[1]
        return [1.0, float(data_scale/data_cores), float(data_cores), float(np.log(data_cores))]

if __name__ == "__main__":
    if len(sys.argv)!= 2:
        print'<usage>  python train_model_and_estimate.py training_points.csv).'
        sys.exit(0)

    test_cores = [2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 104, 128, 152, 192, 256]
    test_scale = 1.0
    test_points = [[i, test_scale] for i in test_cores]

    data = train_model_and_estimate(file_address=sys.argv[1])
    estimation = data.estimate(test_points)

    print
    print "cores, estimated_runtime:"
    for i in range(len(estimation)):
       print test_cores[i], estimation[i]

    plt.plot(test_cores, estimation, 'ro', test_cores, estimation)
    plt.ylabel('Estimated_Runtime')
    plt.xlabel('Num_Cores')
    plt.show()
