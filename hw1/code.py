import numpy as np
import matplotlib.pyplot as plt


def get_a(deg_true):
    """
    Inputs:
    deg_true: (int) degree of the polynomial g

    Returns:
    a: (np array of size (deg_true + 1)) coefficients of polynomial g
    """
    return 5 * np.random.randn(deg_true + 1)


def get_design_mat(x, deg):
    """
    Inputs:
    x: (np.array of size N)
    deg: (int) max degree used to generate the design matrix

    Returns:
    X: (np.array of size N x (deg_true + 1)) design matrix
    """
    X = np.array([x ** i for i in range(deg + 1)]).T
    return X


def draw_sample(deg_true, a, N):
    """
    Inputs:
    deg_true: (int) degree of the polynomial g
    a: (np.array of size deg_true) parameter of g
    N: (int) size of sample to draw

    Returns:
    x: (np.array of size N)
    y: (np.array of size N)
    """
    x = np.sort(np.random.rand(N))
    X = get_design_mat(x, deg_true)
    y = X @ a
    return x, y


def draw_sample_with_noise(deg_true, a, N):
    """
    Inputs:
    deg_true: (int) degree of the polynomial g
    a: (np.array of size deg_true) parameter of g
    N: (int) size of sample to draw

    Returns:
    x: (np.array of size N)
    y: (np.array of size N)
    """
    x = np.sort(np.random.rand(N))
    X = get_design_mat(x, deg_true)
    y = X @ a + np.random.randn(N)
    return x, y




# Q7

def least_square_estimator(X, y):
    assert X.shape[0] >= X.shape[1], 'Bad dimension: N < d: N={}, d={}'.format(X.shape[0], X.shape[1])

    return np.linalg.inv(X.T @ X) @ X.T @ y




# Q8

def empirical_risk(X, y, b):
    ans = 0
    for i in range(len(X)):
        ans += (np.dot(X[i], b) - y[i]) ** 2
    return (1/(2*X.shape[0])) * ans





# Q9

deg_true = 2
a = get_a(deg_true)
x_train, y_train = draw_sample(deg_true, a, 10)
x_test, y_test = draw_sample(deg_true, a, 1000)


def plot_Q9(x_train, y_train, g, f):
    plt.xlim([0, 1])
    plt.plot(x_train, y_train, 'o', label='data_train')
    plt.ylabel('y')
    plt.xlabel('x')
    y_g = [np.polyval(list(reversed(g)), i / 100) for i in range(100)]
    plt.plot([i / 100 for i in range(100)], y_g, label='g(x)')
    y_f = [np.polyval(list(reversed(f)), i / 100) for i in range(100)]
    plt.plot([i / 100 for i in range(100)], y_f, label='f(x)')

    plt.legend(loc="lower left")
    plt.show()


def Q9(d, x_train, y_train, x_test, y_test):
    X_design_mat_train = get_design_mat(x_train, d)
    X_design_mat_test = get_design_mat(x_test, d)

    b_hat = least_square_estimator(X_design_mat_train, y_train)
    print('a equals b_hat (up to deg={}): {}'.format(deg_true, np.allclose(a, b_hat[:deg_true + 1])))
    print('a:{}\nb_hat:{}'.format(a, b_hat))
    plot_Q9(x_train, y_train, a, b_hat)

    return empirical_risk(X_design_mat_train, y_train, b_hat)


d = 5

print('Empirical risk, deg_true={}, d={}: {}'.format(deg_true, d, Q9(d, x_train, y_train, x_test, y_test)))





# Q10

def Q10_helper(deg_true, d, x_train, y_train):
    X_design_mat_train = get_design_mat(x_train, d)

    b_hat = least_square_estimator(X_design_mat_train, y_train)
    plot_Q9(x_train, y_train, a, b_hat)

    return empirical_risk(X_design_mat_train, y_train, b_hat)


def Q10():
    deg_true = 2
    for d in range(1, 8):
        print('Empirical risk, deg_true={}, d={}: {}'.format(deg_true, d, Q10_helper(deg_true, d, x_train, y_train)))


Q10()




# Q11
from random import sample

deg_true = 2
a = get_a(2)

x_test, y_test = draw_sample_with_noise(deg_true, a, 1000)

res_test = {}
res_train = {}

idx_list = [i for i in range(999)]

for d in [2, 5, 10]:
    res_test[d] = {}
    res_train[d] = {}
    x_train_all, y_train_all = draw_sample_with_noise(deg_true, a, 999)

    for N in range(d + 1, 1000):
        idx = sample(idx_list, N)
        x_train = np.array([x_train_all[i] for i in idx])
        y_train = np.array([y_train_all[i] for i in idx])
        if N in [20, 100, 500]:
            Q10_helper(deg_true, d, x_train, y_train)

        b_hat = least_square_estimator(get_design_mat(x_train, d), y_train)

        res_train[d][N] = empirical_risk(get_design_mat(x_train, d), y_train, b_hat)
        res_test[d][N] = empirical_risk(get_design_mat(x_test, d), y_test, b_hat)



plt.figure(figsize=(20,10))
plt.yscale('log')
plt.ylabel('Errors (log scale)')
plt.xlabel('N')
for k in res_test.keys():
    x1, y1 = [int(i) for i in res_test[k].keys()], [float(i) for i in res_test[k].values()]
    x2, y2 = [int(i) for i in res_train[k].keys()], [float(i) for i in res_train[k].values()]
    plt.plot(x1, y1, label='test d={}'.format(k))
    plt.plot(x2, y2, label='train d={}'.format(k))

plt.legend(loc="upper right")
plt.show()





# Q12

q12_res = {}

x_train_all, y_train_all = draw_sample_with_noise(deg_true, a, 999)
x_test, y_test = draw_sample_with_noise(deg_true, a, 1000)


for d in [2, 5, 10]:
    q12_res[d] = {}
    for N in range(d + 1, 1000):
        idx = sample(idx_list, N)
        x_train = np.array([x_train_all[i] for i in idx])
        y_train = np.array([y_train_all[i] for i in idx])

        b_hat = least_square_estimator(get_design_mat(x_train, d), y_train)
        a_hat = np.zeros(b_hat.shape[0])
        a_hat[:a.shape[0]] = a
        q12_res[d][N] = empirical_risk(get_design_mat(x_test, d), y_test, b_hat) - empirical_risk(
            get_design_mat(x_test, 2), y_test, a)

plt.figure(figsize=(20, 10))
plt.ylabel('Estimation Errors')
plt.xlabel('N')
plt.ylim([-1, 10])
for d in q12_res.keys():
    print(d)
    x1, y1 = [int(i) for i in q12_res[d].keys()], [float(i) for i in q12_res[d].values()]
    plt.plot(x1, y1, label='d={}'.format(d))

plt.legend(loc="upper right")
plt.show()