from __future__ import division
import numpy as np

try:
    xrange
except NameError:
    xrange = range

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y

def calc_grad(X, Y, theta):
    # print(str(theta))
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    # add norm of theta - shouldn't this reduce the norm?
    grad = grad + 0.2 * theta

    return grad

def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    # scaling X 10 times
    # X = X*10
    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta  - learning_rate * (grad)
        # theta = theta - learning_rate/np.square(i) * (grad)

        norm = np.linalg.norm(prev_theta - theta)
        if i % 10000 == 0:
            print('Finished %d iterations' % i)

        if i % 1000 == 0:
            print('norm of theta: ' + str(np.linalg.norm(theta)))
            print('norm of delta: ' + str(norm))

        if norm < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return

def main():
    # print('==== Training model on data set A ====')
    # Xa, Ya = load_data('data_a.txt')
    # logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = load_data('data_b.txt')
    logistic_regression(Xb, Yb)

    return

if __name__ == '__main__':
    main()