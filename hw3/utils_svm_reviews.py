import os
import numpy as np
import random
from collections import Counter
import time
import matplotlib.pyplot as plt


def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings.
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on',
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(str.maketrans("", "", symbols)).strip(), lines)
    words = filter(None, words)
    return list(words)


def load_and_shuffle_data():
    '''
    pos_path is where you save positive review data_reviews.
    neg_path is where you save negative review data_reviews.
    '''
    pos_path = "data_reviews/pos"
    neg_path = "data_reviews/neg"

    pos_review = folder_list(pos_path,1)
    neg_review = folder_list(neg_path,-1)

    review = pos_review + neg_review
    random.shuffle(review)
    return review

# Taken from http://web.stanford.edu/class/cs221/ Assignment #2 Support Code
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale


# Q5
def sparse_representation(list_of_words):
    return Counter(list_of_words)


# Q6
def train_test_split():
    data = load_and_shuffle_data()
    X_train = data[:1500]
    y_train = [l[-1] for l in X_train]
    X_train = [sparse_representation(l[:-1]) for l in X_train]

    X_test = data[1500:]
    y_test = [l[-1] for l in X_test]
    X_test = [sparse_representation(l[:-1]) for l in X_test]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = train_test_split()

# Q10


def classification_error(w, X, y):
    # y_pred = []
    # for x in X:
    #     y_pred.append(sign(dotProduct(w, x)))
    #
    # return sum([y[i] != y_pred[i] for i in range(len(y))]) / len(y)

    res = 0
    for i in range(len(y)):
        if y[i] * dotProduct(X[i], w) > 0:
            res += 1
    return 1 - res / len(y)


# Q7

def Q7_pegasos(X_train, y_train, lamb, epoch=10, shuffle=False, verbose=False, eval_step=10):
    assert lamb > 0

    X, Y = X_train, y_train
    w, t, dim = {}, 1, len(Y)

    for i in range(epoch):
        if shuffle:
            temp = list(zip(X, Y))
            random.shuffle(temp)
            X, Y = zip(*temp)
            del temp

        for j in range(dim):
            x, y = X[j], Y[j]
            t += 1
            eta = 1 / (t * lamb)
            coef = 1 - eta * lamb

            for k, v in w.items():
                w[k] = coef * v

            if y * dotProduct(w, x) < 1:
                for k, v in x.items():
                    w[k] = w.get(k, 0) + v * eta * y

        if verbose and not i % eval_step:
            print('epoch: {} with classification error: {}'.format(i, classification_error(w, X, Y)))

    return w


# Hyperparams
epoch = 20
lamb = .5
shuffle = False

# start = time.time()
# Q7_ans = Q7_pegasos(X_train=X_train, y_train=y_train, lamb=lamb, epoch=epoch, shuffle=shuffle)
# end = time.time()
# print('Pegasos before optimization with epoch={}: {}s'.format(epoch, end - start))

# Q8

def Q8_pegasos(X_train, y_train, lamb, epoch=10, shuffle=False, verbose=False, eval_step = 10):
    assert lamb > 0

    X, Y = X_train, y_train
    w, s, t, dim = {}, 1, 1, len(Y)

    for i in range(epoch):
        if shuffle:
            temp = list(zip(X, Y))
            random.shuffle(temp)
            X, Y = zip(*temp)
            del temp

        for j in range(dim):
            x, y = X[j], Y[j]
            t += 1
            eta = 1 / (t * lamb)
            s *= 1 - eta * lamb

            if y * dotProduct(w, x) < 1:
                for k, v in x.items():
                    w[k] = w.get(k, 0) + v * eta * y / s

        if verbose and not i % eval_step:
            res = {}
            increment(res, s, w)
            print('epoch: {} with classification error: {}'.format(i, classification_error(res, X, Y)))

    res = {}
    increment(res, s, w)

    return res


# start = time.time()
# Q8_ans = Q8_pegasos(X_train=X_train, y_train=y_train, lamb=lamb, epoch=epoch, shuffle=shuffle)
# end = time.time()
# print('Pegasos after optimization with epoch={}: {}s'.format(epoch, end - start))



# Q11

# lambdas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10]
# model_errors = []
# for lamb in lambdas:
#     w = Q8_pegasos(X_train=X_train, y_train=y_train, lamb=lamb, epoch=20, shuffle=True)
#     model_errors.append(classification_error(w, X_test, y_test))
# print(model_errors)
#
# print('Best lambda = {}, with classification error = {}'.format(lambdas[model_errors.index(min(model_errors))], min(model_errors)))
#
# plt.xscale('log')
# plt.plot(lambdas, model_errors)
# plt.ylabel('Classification Error')
# plt.xlabel('Lambda')
# plt.title('Q11: classification error on test set w.r.t. lambda')
# plt.savefig('figures/Q11')
# plt.show()

# Q12
w = Q8_pegasos(X_train=X_train, y_train=y_train, lamb=1e-5, epoch=20, shuffle=True)
y_pred = [dotProduct(w, xi) for xi in X_test]

y_ypred = [(y_test[i], y_pred[i]) for i in range(len(y_pred))]

bins = [float('-inf'), -10000, -1000, -100, 0, 100, 1000, 10000, float('inf')]

stat_false, stat_count = [0 for _ in range(len(bins))], [0 for _ in range(len(bins))]

c = 0
res = []
for i, (y, yp) in enumerate(y_ypred):
    if c == 2:
        break

    if not y * yp > 0:
        print(y, yp)
        res.append(i)
        c += 1

import pandas as pd
pd.set_option('display.max_rows', None)


for i in res:
    df = []
    print('*' * 50)
    x = X_test[i]
    for k, v in x.items():
        weight = w[k] if k in w else 0
        df.append((k, v, weight, v * weight))

    df = pd.DataFrame(df)
    df.columns = ['feature_name', 'feature_value', 'feature_weight', 'product']
    df.sort_values(inplace=True, by='product', key= lambda x: abs(x), ascending=False)
    print(df)