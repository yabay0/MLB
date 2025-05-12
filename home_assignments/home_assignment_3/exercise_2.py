import numpy as np
import matplotlib.pyplot as plt
import keras


def load_mnist_digits(digits=(3, 8)):
    """
    Load MNIST, filter for two digits, flatten and normalize.
    Returns X_train, y_train.
    """
    (X_tr, y_tr), _ = keras.datasets.mnist.load_data()
    n = X_tr.shape[0]
    X_tr = X_tr.reshape(n, -1).astype("float32") / 255.0

    mask = np.isin(y_tr, digits)
    X_tr, y_tr = X_tr[mask], y_tr[mask]
    y_tr = (y_tr == digits[1]).astype("float32")
    return X_tr, y_tr


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def compute_loss_and_grad(w, b, X, y):
    n = X.shape[0]
    z = X.dot(w) + b
    p = sigmoid(z)
    ce = -np.mean(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8))
    reg = 0.5 * (w.dot(w) + b**2)
    grad_w = X.T.dot(p - y) / n + w
    grad_b = np.mean(p - y) + b
    return ce + reg, grad_w, grad_b


def train_gd(X, y, lr=0.001, iters=1000, rec=10):
    w, b = np.zeros(X.shape[1]), 0.0
    losses = []
    for t in range(1, iters + 1):
        loss, gw, gb = compute_loss_and_grad(w, b, X, y)
        w -= lr * gw
        b -= lr * gb
        if t % rec == 0:
            losses.append(loss)
    return np.array(losses)


def train_sgd(X, y, lr=0.001, iters=1000, batch_size=10, rec=10):
    n = X.shape[0]
    w, b = np.zeros(X.shape[1]), 0.0
    losses = []
    for t in range(1, iters + 1):
        idx = np.random.choice(n, batch_size, replace=False)
        Xb, yb = X[idx], y[idx]
        p = sigmoid(Xb.dot(w) + b)
        grad_w = Xb.T.dot(p - yb) / batch_size + w
        grad_b = np.mean(p - yb) + b
        w -= lr * grad_w
        b -= lr * grad_b
        if t % rec == 0:
            loss, _, _ = compute_loss_and_grad(w, b, X, y)
            losses.append(loss)
    return np.array(losses)
