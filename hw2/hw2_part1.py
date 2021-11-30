import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data():
    # Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('ridge_regression_dataset.csv', delimiter=',')
    X = df.values[:, :-1]
    y = df.values[:, -1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = load_data()


# Q1
def feature_normalization(train, test):
    # Remove constant features, using statistics of training set
    train = train[:, ~np.all(train[1:] == train[:-1], axis=0)]
    test = test[:, ~np.all(train[1:] == train[:-1], axis=0)]

    # Normalization with min-max
    max_train = train.max(axis=0)[None, :]
    min_train = train.min(axis=0)[None, :]

    train_normalized = (train - min_train) / (max_train - min_train)
    test_normalized = (test - min_train) / (max_train - min_train)

    return train_normalized, test_normalized


# Q5
def compute_square_loss(X, y, theta):
    m = y.shape[0]
    y = y.reshape((y.shape[0], 1))

    return (1 / m * (X @ theta - y).T @ (X @ theta - y))[0, 0]

# Q5 Test
theta = np.random.randn(X_train.shape[1])
compute_square_loss(X_train, y_train, theta)


# Q6
def compute_square_loss_gradient(X, y, theta):
    m = y.shape[0]
    y = y.reshape((-1, 1))

    return ((2 / m) * X.T @ (X @ theta - y)).reshape(-1)

# Q6 Test
theta = np.random.randn(X_train.shape[1], 1)
compute_square_loss_gradient(X_train, y_train, theta)


# Q7
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    true_gradient = compute_square_loss_gradient(X, y, theta)  # The true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)  # Initialize the gradient we approximate
    for i in range(num_features):
        h = np.zeros((num_features, 1))
        h[i] = 1
        approx_grad[i] = (compute_square_loss(X, y, theta + epsilon * h)
                          - compute_square_loss(X, y, theta - epsilon * h)) / (2 * epsilon)

    return np.linalg.norm(approx_grad - true_gradient) <= tolerance


# Q7, generic
def generic_gradient_checker(X, y, theta, objective_func, gradient_func,
                             epsilon=0.01, tolerance=1e-4):
    true_gradient = gradient_func(X, y, theta)
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)
    for i in range(num_features):
        h = np.zeros((num_features, 1))
        h[i] = 1
        approx_grad[i] = (objective_func(X, y, theta + epsilon * h)
                          - objective_func(X, y, theta - epsilon * h)) / (2 * epsilon)

    return np.linalg.norm(approx_grad - true_gradient) <= tolerance

# Q7, generic, test

theta = np.random.randn(X_train.shape[1], 1)
generic_gradient_checker(X_train, y_train, theta, compute_square_loss, compute_square_loss_gradient)


# Q8
def batch_grad_descent(X, y, alpha=0.1, num_step=1000, grad_check=False):
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_step + 1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_step + 1)  # Initialize loss_hist
    theta = np.zeros(num_features)  # Initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta.reshape((-1, 1)))

    for i in range(num_step):
        grad = compute_square_loss_gradient(X, y, theta.reshape((-1, 1)))

        # Do not update theta if grad_check fails
        if grad_check and not grad_checker(X, y, theta.reshape((-1, 1))):
            theta_hist[i + 1, :] = theta
            loss_hist[i + 1] = compute_square_loss(X, y, theta.reshape((-1, 1)))
            continue

        theta = theta - alpha * grad.reshape((1, -1))
        theta_hist[i + 1, :] = theta
        loss_hist[i + 1] = compute_square_loss(X, y, theta.reshape((-1, 1)))

    return theta_hist, loss_hist


# Q9

plt.ylim([0, 10])
plt.xlim([0, 1000])
plt.ylabel('Avg square loss, train set')
plt.xlabel('Epoch')

for alpha in [.01, .05]:
    _, l = batch_grad_descent(X_train, y_train, alpha=alpha, grad_check=False)
    plt.plot([i for i in range(l.shape[0])], l, label='alpha={}'.format(alpha))

plt.legend(loc="upper right")
plt.title('Q9, plot for alpha=0.01/0.05, converge')
plt.savefig('fig/Q9_small_alpha.png')
plt.show()

plt.ylim([0, 10000000])
# plt.yscale('log')
plt.xlim([0, 10])
plt.ylabel('Avg square loss, train set')
plt.xlabel('Epoch')

for alpha in [.1, .5]:
    _, l = batch_grad_descent(X_train, y_train, alpha=alpha, grad_check=False)
    plt.plot([i for i in range(l.shape[0])], l, label='alpha={}'.format(alpha))

plt.legend(loc="lower right")
plt.title('Q9, plot for alpha=0.1/0.5, diverge')
plt.savefig('fig/Q9_big_alpha.png')
plt.show()

# Q10

plt.ylim([2, 6])
plt.xlim([0, 1000])
plt.ylabel('Avg square loss')
plt.xlabel('Epoch')

for alpha in [.01, .05]:
    # Grad descent, training loss
    theta_list, l = batch_grad_descent(X_train, y_train, alpha=alpha, grad_check=False)

    testing_loss = []
    for i in range(theta_list.shape[0]):
        testing_loss.append(compute_square_loss(X_test, y_test, theta_list[i].reshape(-1, 1)))

    plt.plot([i for i in range(len(testing_loss))], testing_loss, label='Test loss, alpha={}'.format(alpha))

plt.legend(loc="upper right")
plt.title('Q10, plot for alpha=0.01/0.05, test set, converge')
plt.savefig('fig/Q10_small_alpha.png')
plt.show()

plt.ylim([0, 10000000])
plt.xlim([0, 10])
plt.ylabel('Avg square loss')
plt.xlabel('Epoch')

for alpha in [.1, .5]:
    # Grad descent, training loss
    theta_list, _ = batch_grad_descent(X_train, y_train, alpha=alpha, grad_check=False)

    testing_loss = []
    for i in range(theta_list.shape[0]):
        testing_loss.append(compute_square_loss(X_test, y_test, theta_list[i, :]))

    plt.plot([i for i in range(len(testing_loss))], testing_loss, label='Test loss, alpha={}'.format(alpha))

plt.legend(loc="upper right")
plt.title('Q10, plot for alpha=0.1/0.5, test set, diverge')
plt.savefig('fig/Q10_big_alpha.png')
plt.show()



# Q12
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    m = y.shape[0]
    y = y.reshape((-1, 1))

    return ((2 / m) * X.T @ (X @ theta - y) + 2 * lambda_reg * theta).reshape(-1)


# Q13
def regularized_grad_descent(X, y, alpha=0.05, lambda_reg=10 ** -2, num_step=1000):
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_features)  # Initialize theta
    theta_hist = np.zeros((num_step + 1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_step + 1)  # Initialize loss_hist
    theta_hist[0, :] = theta
    loss_hist[0] = compute_square_loss(X, y, theta.reshape((-1, 1)))

    for i in range(num_step):
        grad = compute_regularized_square_loss_gradient(X, y, theta.reshape((-1, 1)), lambda_reg)
        theta = theta - alpha * grad.reshape((1, -1))
        theta_hist[i + 1, :] = theta
        loss_hist[i + 1] = compute_square_loss(X, y, theta.reshape((-1, 1)))

    return theta_hist, loss_hist



# Q14, choosing alpha=0.05

from matplotlib.pyplot import figure

# Lambda < 1

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.ylim([2, 8])
plt.xlim([0, 1000])
plt.ylabel('Avg square loss, train/test set')
plt.xlabel('Epoch')

alpha = 0.05
for lambda_reg in [10**-7, 10**-5, 10**-3, 10**-1]:
    theta_list, training_loss = regularized_grad_descent(X_train, y_train, alpha=alpha, lambda_reg=lambda_reg)
    plt.plot([i for i in range(l.shape[0])], training_loss, label='Train, lambda={}'.format(lambda_reg))

    testing_loss = []
    for i in range(theta_list.shape[0]):
        testing_loss.append(compute_square_loss(X_test, y_test, theta_list[i].reshape((-1, 1))))
    plt.plot([i for i in range(l.shape[0])], testing_loss, label='Test, lambda={}'.format(lambda_reg))

plt.legend(loc="upper left")
plt.title('Q14, plot for alpha=0.05 and different λ < 1')
plt.savefig('fig/Q14_alpha=0.05_small_lambda.png')
plt.show()

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.ylim([4, 10000000])
plt.xlim([0, 150])
plt.ylabel('Avg square loss, train/test set')
plt.xlabel('Epoch')

alpha = 0.05
for lambda_reg in [1, 10, 100]:
    theta_list, training_loss = regularized_grad_descent(X_train, y_train, alpha=alpha, lambda_reg=lambda_reg)
    plt.plot([i for i in range(l.shape[0])], training_loss, label='Train, lambda={}'.format(lambda_reg))

    testing_loss = []
    for i in range(theta_list.shape[0]):
        testing_loss.append(compute_square_loss(X_test, y_test, theta_list[i].reshape((-1, 1))))

    plt.plot([i for i in range(l.shape[0])], testing_loss, label='Test, lambda={}'.format(lambda_reg))

plt.legend(loc="upper center")
plt.title('Q14, plot for alpha=0.05 and different λ >= 1')
plt.savefig('fig/Q14_alpha=0.05_big_lambda.png')
plt.show()



# Q15, various lambda

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.ylim([2, 6])
plt.xscale('log')
plt.ylabel('Avg square loss, train/test set')
plt.xlabel('λ')

alpha = 0.05
lambda_list = [10**-7]
while(lambda_list[-1] < 0.5):
    lambda_list.append(lambda_list[-1] * 2)
lambda_list.append(1)

plot_list_train = []
plot_list_test = []

for lambda_reg in lambda_list:
    # Grad descent, training loss
    theta_list, training_loss = regularized_grad_descent(X_train, y_train, alpha=alpha, lambda_reg=lambda_reg)
    plot_list_train.append(training_loss[-1])
    plot_list_test.append(compute_square_loss(X_test, y_test, theta_list[-1].reshape((-1, 1))))

plt.plot(lambda_list, plot_list_train, label='Train loss')
plt.plot(lambda_list, plot_list_test, label='Test loss')

plt.legend(loc="upper center")
plt.title('Q15, plot for alpha=0.05 and different λ')
plt.savefig('fig/Q15_alpha=0.05.png')
plt.show()

# Q16

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.ylim([2, 6])
plt.xscale('log')
plt.ylabel('Avg square loss, train/test set')
plt.xlabel('λ')

alpha = 0.05
lambda_list = [10 ** -7]
while (lambda_list[-1] < 0.5):
    lambda_list.append(lambda_list[-1] * 2)
lambda_list.append(1)

plot_list_train = []
plot_list_test = []
plot_list_test_min = []

for lambda_reg in lambda_list:
    # Grad descent, training loss
    theta_list, training_loss = regularized_grad_descent(X_train, y_train, alpha=alpha, lambda_reg=lambda_reg)
    plot_list_train.append(training_loss[-1])
    plot_list_test.append(compute_square_loss(X_test, y_test, theta_list[-1].reshape((-1, 1))))

    test_min_cur = float('inf')
    for theta in theta_list:
        test_min_cur = min(test_min_cur, compute_square_loss(X_test, y_test, theta.reshape((-1, 1))))
    plot_list_test_min.append(test_min_cur)

plt.plot(lambda_list, plot_list_train, label='Train loss')
plt.plot(lambda_list, plot_list_test, label='Test loss')
plt.plot(lambda_list, plot_list_test_min, label='Test loss minimum')

plt.legend(loc="upper center")
plt.title('Q16, plot for alpha=0.05 and different λ, with min_test_loss')
plt.savefig('fig/Q16_alpha=0.05.png')
plt.show()



