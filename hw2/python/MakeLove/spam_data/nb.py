from __future__ import division
import numpy as np

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    m, n = matrix.shape
    phi_1 = [None] * n
    phi_0 = [None] * n

    ones_count = np.count_nonzero(category)
    zeros_count = len(category) - ones_count

    ones_count_p = ones_count/len(category)
    zeros_count_p = 1 - ones_count_p

    # logP(y=1)
    log_ones = np.log(ones_count_p)
    # logP(y=0)
    log_zeros = np.log(zeros_count_p)

    ###################
    # use the laplace term
    # counts # of training examples where y = 1
    # counts # of training examples where y = 0
    for i in range(1, n):
        if i % 1000 == 0:
            print("training: " + str(i))
        token_i_column = matrix[:, i]

        non_zero_count_for_token_i = 0
        zeros_count_for_token_i = 0
        for j in range(1, m):
            if category[j] == 1 and token_i_column[j] != 0:
                non_zero_count_for_token_i += 1
            if category[j] == 0 and token_i_column[j] != 0:
                zeros_count_for_token_i += 1

        # +1 to avoid log(0)
        phi_0[i] = np.log(zeros_count_for_token_i + 1) - np.log(zeros_count + n)
        phi_1[i] = np.log(non_zero_count_for_token_i + 1) - np.log(ones_count + n)

    ###################
    return phi_1, phi_0, log_ones, log_zeros

def nb_test(matrix, phi_1, phi_0, log_ones, log_zeros):
    m, n = matrix.shape
    output = np.zeros(m)
    ###################
    for i in range(1, m):
        if i % 1000 == 0:
            print("testing: " + str(i))
        token_i_row = matrix[i ,:]
        estimate_log_ones = 0
        estimate_log_zeros = 0
        for j in range(1, n):
            if token_i_row[j] > 0:
                estimate_log_ones += phi_1[j] * token_i_row[j]
                estimate_log_zeros += phi_0[j] * token_i_row[j]

        estimate_log_ones += log_ones
        estimate_log_zeros += log_zeros

        if estimate_log_zeros > estimate_log_ones:
            output[i] = 0
        else:
            output[i] = 1
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print 'Error: %1.4f' % error

def find_bad_tokens(tokenlist, phi_1, phi_0):
    n = len(phi_1)

    output = [None] * n

    for i in range(1, n):
        # note phi are already log, so we just minus them here
        output[i] = phi_1[i] - phi_0[i]

    sorted_indices = sorted(range(n), key=lambda k: output[k], reverse=True)
    for i in range(5):
        print tokenlist[sorted_indices[i]]

def train_set(training_set_name):
    print('training: ' + str(training_set_name))
    trainMatrix, tokenlist, trainCategory = readMatrix(training_set_name)
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    phi_1, phi_0, log_ones, log_zeros = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, phi_1, phi_0, log_ones, log_zeros)

    evaluate(output, testCategory)


    # find_bad_tokens(tokenlist, phi_1, phi_0)
    # result for MATRIX.TRAIN:
        # output:
        # spam
        # httpaddr
        # unsubscrib
        # websit
        # lowest


def main():
    #training_datas = ['MATRIX.TRAIN']
    # output: Error: 0.0262

    training_datas = ['MATRIX.TRAIN.50', 'MATRIX.TRAIN.100', 'MATRIX.TRAIN.200', 'MATRIX.TRAIN.400',
                      'MATRIX.TRAIN.800', 'MATRIX.TRAIN.1400']
    for training_data in training_datas:
        train_set(training_data)

    # training: MATRIX.TRAIN.50
    # Error: 0.3475
    # training: MATRIX.TRAIN.100
    # Error: 0.2062
    # training: MATRIX.TRAIN.200
    # Error: 0.0725
    # training: MATRIX.TRAIN.400
    # Error: 0.0200
    # training: MATRIX.TRAIN.800
    # Error: 0.0213
    # training: MATRIX.TRAIN.1400
    # Error: 0.0238

    return

if __name__ == '__main__':
    main()
