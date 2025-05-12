import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import zero_one_loss


def load_data(path="nonlinear_svm_data.csv"):
    """
    Load dataset and map labels {0,1} -> {-1,+1}, then standardize features.
    """
    df = pd.read_csv(path)
    X = df[["x", "y"]].values
    y = df["label"].map({1: +1, 0: -1}).values
    # standardize to zero mean, unit variance
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def plot_data(X, y, title="Data", fname="exercise1_data.png"):
    """
    Scatter-plot positive vs negative points.
    """
    pos = y == +1
    neg = y == -1

    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=12)
    plt.scatter(X[pos, 0], X[pos, 1], c="blue", marker="+", s=50, label="Positive (+1)")
    plt.scatter(
        X[neg, 0],
        X[neg, 1],
        c="green",
        marker="o",
        s=50,
        edgecolors="k",
        label="Negative (-1)",
    )
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def run_linear_svm(X, y, C_values=(1, 100, 1000)):
    """
    Train LinearSVC for each C and print zero-one loss.
    """
    print("Linear SVM (zero-one loss):")
    for C in C_values:
        clf = LinearSVC(C=C, loss="hinge", max_iter=10000)
        clf.fit(X, y)
        preds = np.sign(clf.decision_function(X))
        loss = zero_one_loss(y, preds)
        print(f"  C={C:<4d} -> loss = {loss:.4f}")


def run_rbf_svm(X, y, a_values=(0.1, 1, 10), C=1):
    """
    Train RBF-kernel SVC for each a and print zero-one loss.
    """
    print("\nRBF-kernel SVM (C=1):")
    for a in a_values:
        gamma = 1.0 / (2 * a**2)
        clf = SVC(C=C, kernel="rbf", gamma=gamma)
        clf.fit(X, y)
        loss = zero_one_loss(y, clf.predict(X))
        print(f"  a={a:<4.1f} -> loss = {loss:.4f}")
