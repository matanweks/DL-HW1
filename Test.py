
import matplotlib.pyplot as plt
import numpy as np

import pickle as pickle
import gzip as gzip
import urllib.request as urllib_request
import pickle as pickle
import json as json


data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
# Load the dataset
# urllib_request.urlretrieve(data_url, "mnist.pkl.gz")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')


def ex_a():
    train_mean = np.mean(train_set[0], axis=0)
    train_set_pre_process = train_set[0] - train_mean
    valid_set_pre_process = valid_set[0] - train_mean
    test_set_pre_process = test_set[0] - train_mean


ex_a()




# #
#
# # HW 1  Avishag and Matan
# # Part A:
#
#
# # Plot test
# # def function
# x = np.linspace(0, 784, 784)
# y = train_set
# fig = plt.figure()
#
# ax1 = fig.gca()
# lines1 = ax1.plot(x1, y1, 'r*')
#
# plt.show()
#
