#
# import matplotlib.pyplot as plt
# import numpy as np
# import pickle, gzip, import urllib.request, json
#
# data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
# # Load the dataset
# urllib_request.urlretrieve(data_url, "mnist.pkl.gz")
# with gzip.open('mnist.pkl.gz', 'rb') as f:
#     train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
#
# # section A functions
#
# def find_mean(data_set):
#     train_mean = np.mean(train_set[0], axis=0)
#     return train_mean
#
# def preprocess(data_set, train_mean):
#     data_set_processed = data_set - train_mean
#     return data_set_processed
#
# # section B functions
# def analytic_regression(x,y):
#     w_analytic = np.linalg.pinv(X.T * X + 2 * X.shape[0] * c * np.identity(X.shape[1])) * (X.T * y)
#     b_analytic = 1 / y.shape[0] * np.sum(y)
#     return w_analytic, b_analytic
#
# def gradient_descent_regression():
#     for()
# # extra
# def plt_the_number(x):
#     train_set_test1 = np.reshape(x, (28, 28))
#     plt.imshow(train_set_test1, cmap='gray')
#     plt.show()
#     # x = np.linspace(0, 784, 784)
#     # y = train_set[0, 1]
#     # fig = plt.figure()
#     # ax1 = fig.gca()
#     # lines1 = ax1.plot(x, y, 'r*')
#     # plt.show()
#
# ######################## MAIN ########################
# # section A
# train_mean = find_mean(train_set[0])
# train_set_process = preprocess(train_set[0], train_mean)
# valid_set_process = preprocess(valid_set[0], train_mean)
# test_set_process = preprocess(test_set[0], train_mean)
#
# # section B
# [w, x] = analytic_regression(train_set_process, )
#

for x in range(0, 100):
    print(x)
