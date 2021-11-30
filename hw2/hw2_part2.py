import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def pre_process_mnist_01():
    """
    Load the mnist datasets, selects the classes 0 and 1
    and normalize the data.
    Args: none
    Outputs:
        X_train: np.array of size (n_training_samples, n_features)
        X_test: np.array of size (n_test_samples, n_features)
        y_train: np.array of size (n_training_samples)
        y_test: np.array of size (n_test_samples)
    """
    X_mnist, y_mnist = fetch_openml('mnist_784', version=1,
                                    return_X_y=True, as_frame=False)
    indicator_01 = (y_mnist == '0') + (y_mnist == '1')
    X_mnist_01 = X_mnist[indicator_01]
    y_mnist_01 = y_mnist[indicator_01]
    X_train, X_test, y_train, y_test = train_test_split(X_mnist_01, y_mnist_01,
                                                        test_size=0.33,
                                                        shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_test = 2 * np.array([int(y) for y in y_test]) - 1
    y_train = 2 * np.array([int(y) for y in y_train]) - 1
    return X_train, X_test, y_train, y_test


def sub_sample(N_train, X_train, y_train):
    """
    Subsample the training data to keep only N first elements
    Args: none
    Outputs:
        X_train: np.array of size (n_training_samples, n_features)
        X_test: np.array of size (n_test_samples, n_features)
        y_train: np.array of size (n_training_samples)
        y_test: np.array of size (n_test_samples)
    """
    assert N_train <= X_train.shape[0]
    return X_train[:N_train, :], y_train[:N_train]

X_train, X_test, y_train, y_test = pre_process_mnist_01()


# Q25

def classification_error(clf, X, y):
    return np.sum(clf.predict(X) != y) / y.shape[0]


# Q26
from collections import defaultdict

X_train, y_train = sub_sample(100, X_train, y_train)

q26_res = defaultdict(list)

for alpha in [0.0001, 0.0003, 0.0007, 0.001, 0.003, 0.007, 0.01, 0.03, 0.07, 0.1]:
    for _ in range(10):
        clf = SGDClassifier(loss='log', max_iter=1000,
                            tol=1e-3,
                            penalty='l1', alpha=alpha,
                            learning_rate='invscaling',
                            power_t=0.5,
                            eta0=0.01,
                            verbose=1)
        clf.fit(X_train, y_train)

        q26_res[alpha].append(classification_error(clf, X_test, y_test))


q26_plot = [(alpha, np.mean(val), np.std(val)) for alpha, val in q26_res.items()]

plt.xscale('log')
plt.ylabel('Classification Error')
plt.xlabel('α')

plt.errorbar([x[0] for x in q26_plot], [x[1] for x in q26_plot], yerr=[x[2] for x in q26_plot], uplims = True, lolims = True,)

plt.title('Q26, classification error for logit regression')
plt.savefig('fig/Q26.png')
plt.show()


# Q29
X_train, y_train = sub_sample(100, X_train, y_train)

fig, axs = plt.subplots(2, 5, figsize=(24, 8))

for idx, alpha in enumerate([0.0001, 0.0003, 0.0007, 0.001, 0.003, 0.007, 0.01, 0.03, 0.07, 0.1]):
    clf = SGDClassifier(loss='log', max_iter=1000,
                            tol=1e-3,
                            penalty='l1', alpha=alpha,
                            learning_rate='invscaling',
                            power_t=0.5,
                            eta0=0.01,
                            verbose=1)
    clf.fit(X_train, y_train)

    theta = clf.coef_.reshape((28, 28))
    scale = np.abs(theta).max()
    im = axs[idx//5, idx%5].imshow(theta, cmap=plt.cm.RdBu, vmax=scale, vmin=-scale)
    axs[idx//5, idx%5].title.set_text("Alpha={}".format(alpha))
    plt.colorbar(im, ax=axs[idx//5, idx%5], fraction=0.046, pad=0.04)

plt.savefig('fig/Q29.png')