import matplotlib.pyplot as plt
import numpy as np
import pickle, gzip, urllib.request, json

# Load the dataset
data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
# urllib_request.urlretrieve(data_url, "mnist.pkl.gz")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

# section A functions


def find_mean(data_set):
    train_mean = np.mean(data_set, axis=0)
    return train_mean


def preprocess(data_set, train_mean):
    data_set_processed = data_set - train_mean
    return data_set_processed


def class_process(y):
    y = [(1 if (y[i]%2 == 0) else -1) for i in range(len(y))]
    y = np.array(y)
    return y


# section B functions
def analytic_regression(x, y, lambda1):
    m = x.shape[0]
    b_analytic = 1 / m * np.sum(y)
    w_analytic = np.matmul(np.linalg.pinv(np.matmul(x.T, x) + 2 * m * lambda1 * np.identity(x.shape[1])), np.matmul(x.T, y))
    return w_analytic, b_analytic


def gradient_descent_regression(x, y, lambda1, w_new, b_new, step_size=0.001):
    y = convert_vector_to_matrix(y)
    m = x.shape[0]
    for i in range(0, 100):
        w_old = w_new
        x_mul_w_minus_y = np.matmul(x, w_old) - y
        w_derivative = 1/m * (np.matmul(x.T, x_mul_w_minus_y)) + 2 * lambda1 * w_old
        w_new = w_old - step_size * w_derivative

        b_old = b_new
        b_new = b_old - step_size * (b_old - 1/m * np.sum(y))
    return w_new, b_new

# section C functions
# def linear_classifier():

# extras
def plt_the_number(y):
    reshape_set = np.reshape(y, (28, 28))
    plt.imshow(reshape_set, cmap='gray')
    plt.show()

    x = np.linspace(0, y.shape[0], y.shape[0])
    # y = train_set[0, 1]
    fig = plt.figure()
    ax1 = fig.gca()
    lines1 = ax1.plot(x, y, 'r*')
    plt.show()


def plt_w(w_analytic1, w_gd1):

    x = np.linspace(0, w_analytic1.shape[0], w_analytic1.shape[0])
    fig = plt.figure()
    ax1 = fig.gca()
    line1, line2 = ax1.plot(x, w_analytic1, 'r*', x, w_gd1, 'g*')
    plt.show()


def print_w_b(w_analytic, b_analytic):
    print("w_analytic:")
    print("shape:")
    print(w_analytic.shape)
    print("b_analytic:")
    print(b_analytic)


def convert_vector_to_matrix(y):
    y = y[:, np.newaxis]
    return y


######################## MAIN ########################

def main():
    # section A
    train_mean = find_mean(train_set[0])
    train_set_process = preprocess(train_set[0], train_mean)
    valid_set_process = preprocess(valid_set[0], train_mean)
    test_set_process = preprocess(test_set[0], train_mean)
    train_set_class = class_process(train_set[1])

    # train_set_test = train_set[0][6]
    # plt_the_number(train_set_test)
    # plt_the_number(train_mean)
    # plt_the_number(test_set_process[6])

    lambda1 = 0.001
    # initial_guess = np.zeros((784, 1))
    initial_guess = np.random.rand(784, 1)

    # section B
    [w_analytic, b_analytic] = analytic_regression(train_set_process, train_set_class, lambda1)
    [w_gd, b_gd] = gradient_descent_regression(x=train_set_process, y=train_set_class, lambda1=lambda1, w_new=initial_guess, b_new=0)
    # print_w_b(w_gd, b_gd)
    plt_w(w_analytic, w_gd)
    # print("w_analytic")
    # print(w_analytic)
    # print("w_gd")
    # print(w_gd)

    # section C

main()








