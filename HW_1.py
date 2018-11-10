import matplotlib.pyplot as plt
import numpy as np
import pickle, gzip, urllib.request, json

# Load the dataset
data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
# urllib_request.urlretrieve(data_url, "mnist.pkl.gz")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

############################################################################################################

# section A functions

def find_mean(data_set):
    train_mean = np.mean(data_set, axis=0)
    return train_mean

############################################################################################################

def preprocess(data_set, train_mean):
    data_set_processed = data_set - train_mean
    return data_set_processed

############################################################################################################

def class_process(y):
    y = [(1 if (y[i]%2 == 0) else -1) for i in range(len(y))]
    y = np.array(y)
    return y

############################################################################################################

# section B functions
def analytic_regression(x, y, lambda1):
    m = x.shape[0]
    b_analytic = 1 / m * np.sum(y)
    w_analytic = np.matmul(np.linalg.pinv(np.matmul(x.T, x) + 2 * m * lambda1 * np.identity(x.shape[1])), np.matmul(x.T, y))
    return w_analytic, b_analytic

############################################################################################################

def GD_regression(x, y, lambda1, w_new, b_new, step_size=0.01, calc_loss = 0, x_test = 0, y_test = 0):
    y = convert_vector_to_matrix(y)
    m = x.shape[0]
    iterations = 100
    loss_01 = np.zeros([iterations,2])
    sqloss = np.zeros([iterations,2])
    for i in range(0, iterations):
       # print(i)
        w_old = w_new
        x_mul_w_minus_y = np.matmul(x, w_old) - y
        w_derivative = 1/m * (np.matmul(x.T, x_mul_w_minus_y)) + 2 * lambda1 * w_old
        w_new = w_old - step_size * w_derivative

        b_old = b_new
        b_new = b_old - step_size * (b_old - 1/m * np.sum(y))

        if calc_loss:
            loss_01[i][0] = loss01(x=x, y=y, w=w_new, b=b_new, conversion_needed=0)
            sqloss[i][0] = square_loss(x=x, y=y, w=w_new, b=b_new, conversion_needed=0)
            loss_01[i][1] = loss01(x=x_test, y=y_test, w=w_new, b=b_new)
            sqloss[i][1] = square_loss(x=x_test, y=y_test, w=w_new, b=b_new)

    if calc_loss:
        return w_new, b_new ,loss_01, sqloss, iterations
    else:
        return w_new, b_new

######################################   Section C functions   #############################################


def linear_classifier(w, b, x):
    return np.sign(np.dot(x, w)+b)

def loss01(x, y, w, b, conversion_needed = 1):
    if conversion_needed:
        y = convert_vector_to_matrix(y)
    loss = (0.5 / x.shape[0]) * np.sum(np.abs((linear_classifier(w, b, x) - y)))
    return loss

def square_loss(x, y, w, b, conversion_needed = 1):
    if conversion_needed:
        y = convert_vector_to_matrix(y)
    loss = (1/x.shape[0]) * np.sum((x @ w + b - y)**2)
    return loss


###########################################   EXTRAS   ######################################################

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

# def testing():
#     # train_set_test = train_set[0][6]
#     # plt_the_number(train_set_test)
#     # plt_the_number(train_mean)
#     # plt_the_number(test_set_process[6])

######################################################## MAIN ########################################################

def main():
    #--------------------------------------- section A ---------------------------------------

    train_mean = find_mean(train_set[0])
    train_set_process = preprocess(train_set[0], train_mean)
    valid_set_process = preprocess(valid_set[0], train_mean)
    test_set_process = preprocess(test_set[0], train_mean)
    train_set_class = class_process(train_set[1])
    valid_set_class = class_process(valid_set[1])
    test_set_class = class_process(test_set[1])


    initial_guess = np.zeros((784, 1))

    min_loss_analytic = 1
    min_loss_GD = 1
    lamda_best_analytic = 0
    lamda_best_GD = 0

    for k in range(-5, 3):

        lambda1 = 10**k
        [w_analytic, b_analytic] = analytic_regression(train_set_process, train_set_class, lambda1)
        [w_gd, b_gd] = GD_regression(x=train_set_process, y=train_set_class, lambda1=lambda1, w_new=initial_guess, b_new=-0.01)

        # losses calculations

        loss_analytic_train =  loss01(x=train_set_process, y=train_set_class, w = convert_vector_to_matrix(w_analytic), b = b_analytic)
        sq_loss_analytic_train = square_loss(x=train_set_process, y=train_set_class, w = convert_vector_to_matrix(w_analytic), b = b_analytic)

        loss_analytic_vald = loss01(x=valid_set_process, y=valid_set_class, w=convert_vector_to_matrix(w_analytic), b=b_analytic)
        sq_loss_analytic_vald = square_loss(x=valid_set_process, y=valid_set_class, w=convert_vector_to_matrix(w_analytic), b=b_analytic)

        loss_gd_train =  loss01(x=train_set_process, y=train_set_class, w = w_gd, b = b_gd)
        sq_loss_gd_train = square_loss(x=train_set_process, y=train_set_class, w = w_gd, b = b_gd)

        loss_gd_vald =  loss01(x=valid_set_process, y=valid_set_class, w = w_gd, b = b_gd)
        sq_loss_gd_vald = square_loss(x=valid_set_process, y=valid_set_class, w = w_gd, b = b_gd)

        print("Training Set:")
        print("------------- Lambda =  " + str(lambda1) + "-------------")
        print("Analytic: 0/1 Loss = " + str(loss_analytic_train) + ', Square loss = ' + str(sq_loss_analytic_train))
        print("Gradient Decent: 0/1 Loss = " + str(loss_gd_train) + ', Square loss = ' + str(sq_loss_gd_train))
        print(" ")
        print("Validation Set:")
        print("------------- Lambda =  " + str(lambda1) + "-------------")
        print("Analytic: 0/1 Loss = " + str(loss_analytic_vald) + ', Square loss = ' + str(sq_loss_analytic_vald))
        print("Gradient Decent: 0/1 Loss = " + str(loss_gd_vald) + ', Square loss = ' + str(sq_loss_gd_vald))
        print(" ")

        # find which lambda works best for each scheme

        if loss_analytic_vald < min_loss_analytic:
            min_loss_analytic = loss_analytic_vald
            lamda_best_analytic = lambda1
            w_best_analytic = w_analytic
            b_best_analytic = b_analytic

        if loss_gd_vald < min_loss_GD:
            min_loss_GD = loss_gd_vald
            lamda_best_GD = lambda1
            w_best_GD = w_gd
            b_best_GD = b_gd

    print("Best Lambda for analytic: " +str(lamda_best_analytic))
    print("Best Lambda for GD: " +str(lamda_best_GD))
    print(" ")

    # Section D: Run on test set to get loss

    loss_analytic_test = loss01(x=test_set_process, y=test_set_class, w=convert_vector_to_matrix(w_best_analytic), b=b_best_analytic)
    sq_loss_analytic_test = square_loss(x=test_set_process, y=test_set_class, w=convert_vector_to_matrix(w_best_analytic), b=b_best_analytic)

    loss_gd_test = loss01(x=test_set_process, y=test_set_class, w=w_best_GD, b=b_best_GD)
    sq_loss_gd_test = square_loss(x=test_set_process, y=test_set_class, w=w_best_GD, b=b_best_GD)

    print("Test Set:")
    print("------------- For the best Lambda found (different for each scheme) -------------")
    print("Analytic: Lambda = " + str(lamda_best_analytic) + ", 0/1 Loss = " + str(loss_analytic_test) + ', Square loss = ' + str(sq_loss_analytic_test))
    print("Gradient Decent: Lambda = " + str(lamda_best_GD) + ", 0/1 Loss = " + str(loss_gd_test) + ', Square loss = ' + str(sq_loss_gd_test))
    print(" ")


    # Section E: learning curves

    print("Starting section E")
    print(" ")
    [w_gd, b_gd, loss_01, sqloss, iterations] = GD_regression(x=train_set_process, y=train_set_class, lambda1=lamda_best_GD, w_new=initial_guess, b_new=-0.01, calc_loss=1, x_test=test_set_process, y_test=test_set_class)
    x_axis = np.linspace(1,iterations, iterations)

    plt.plot(x_axis, loss_01[:,0],'r', x_axis, loss_01[:, 1],'b')
    plt.ylabel("0/1 Loss")
    plt.xlabel("Iteration")
    plt.title("0/1 Loss Vs. GD iterations")
    plt.legend(["Training set", "Test set"])
    plt.show()

    plt.plot(x_axis, sqloss[:,0],'r', x_axis, sqloss[:, 1],'b')
    plt.ylabel("Squared Loss")
    plt.xlabel("Iteration")
    plt.title("Squared Loss Vs. GD iterations")
    plt.legend(["Training set", "Test set"])
    plt.show()

main()
