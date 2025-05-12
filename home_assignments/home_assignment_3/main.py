import numpy as np
from exercise_1 import load_data, plot_data, run_linear_svm, run_rbf_svm
from exercise_2 import (
    load_mnist_digits,
    sigmoid,
    compute_loss_and_grad,
    train_gd,
    train_sgd,
)
import matplotlib.pyplot as plt


def run_exercise1():
    """
    Exercise 1: SVM on nonlinear data.
    Major steps:
      1. Load & standardize data
      2. Scatter-plot points (+1 vs –1)
      3. Train Linear SVMs for C = 1,100,1000 and report 0-1 loss
      4. Train RBF-kernel SVMs for a = 0.1,1,10 (c=1) and report loss
    """
    # 1) Load & preprocess
    X, y = load_data(path="nonlinear_svm_data.csv")

    # 2) Plot data distribution
    plot_data(
        X, y, title="Data points (positive: +1, negative: -1)", fname="ex1_data.png"
    )

    # 3) Linear SVM experiments
    run_linear_svm(X, y, C_values=(1, 100, 1000))

    # 4) RBF-kernel SVM experiments
    run_rbf_svm(X, y, a_values=(0.1, 1, 10), C=1)


def run_exercise2():
    """
    Exercise 2: Logistic regression on MNIST digits {3,8}.
    Major steps:
      Q3: GD vs SGD training loss (γ=0.001)
      Q4: Learning-rate comparison (γ ∈ {0.0001, 0.001, 0.01})
      Q5: Batch-size impact (b ∈ {1,10,100}) at best γ
      Q6: Constant vs diminishing step-size (γ_t=1/t)
    """
    # Parameters
    iterations = 1000
    rec = 10
    steps = np.arange(rec, iterations + 1, rec)
    best_gamma = 1e-2

    # Load MNIST and filter digits
    X, y = load_mnist_digits(digits=(3, 8))

    # Q3: GD vs mini-batch SGD (γ=0.001)
    gd_loss = train_gd(X, y, lr=0.001, iters=iterations, rec=rec)
    sgd_loss = train_sgd(X, y, lr=0.001, iters=iterations, batch_size=10, rec=rec)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, gd_loss, marker="o", linestyle="-", linewidth=2, label="GD loss")
    plt.plot(
        steps,
        sgd_loss,
        marker="s",
        linestyle="-",
        linewidth=2,
        label="Mini-batch SGD loss",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Q3: GD vs Mini-batch SGD Loss (γ=0.001)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("ex2_q3_loss.png", dpi=300)
    plt.close()

    # Q4: Learning-rate comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].set_title("Q4: GD Training Loss (various γ)")
    for g in [1e-4, 1e-3, 1e-2]:
        axes[0].plot(
            steps,
            train_gd(X, y, lr=g, iters=iterations, rec=rec),
            marker="o",
            linestyle="-",
            linewidth=2,
            label=f"γ={g}",
        )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Training Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("Q4: Mini-batch SGD Training Loss (various γ)")
    for g in [1e-4, 1e-3, 1e-2]:
        axes[1].plot(
            steps,
            train_sgd(X, y, lr=g, iters=iterations, batch_size=10, rec=rec),
            marker="s",
            linestyle="--",
            linewidth=2,
            label=f"γ={g}",
        )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Training Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("ex2_q4_gamma.png", dpi=300)
    plt.close()

    # Q5: Batch-size impact at best γ
    plt.figure(figsize=(10, 6))
    for b, mk in zip([1, 10, 100], ["o", "s", "D"]):
        loss_b = train_sgd(X, y, lr=best_gamma, iters=iterations, batch_size=b, rec=rec)
        plt.plot(steps, loss_b, marker=mk, linestyle="-", linewidth=2, label=f"b={b}")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title(f"Q5: Mini-batch SGD Loss vs Batch Size (γ={best_gamma})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("ex2_q5_batch.png", dpi=300)
    plt.close()

    # Q6: Constant vs diminishing learning rate (γ_t = 1/t)
    # constant learning rate loss
    const_loss = train_sgd(
        X, y, lr=best_gamma, iters=iterations, batch_size=10, rec=rec
    )
    # diminishing learning rate loss
    w, b0 = np.zeros(X.shape[1]), 0.0
    dimi_losses = []
    for t in range(1, iterations + 1):
        lr_t = 1.0 / t
        idx = np.random.choice(X.shape[0], 10, replace=False)
        Xb, yb = X[idx], y[idx]
        p = sigmoid(Xb.dot(w) + b0)
        gw = Xb.T.dot(p - yb) / 10 + w
        gb = np.mean(p - yb) + b0
        w -= lr_t * gw
        b0 -= lr_t * gb
        if t % rec == 0:
            loss, _, _ = compute_loss_and_grad(w, b0, X, y)
            dimi_losses.append(loss)

    plt.figure(figsize=(10, 6))
    plt.plot(
        steps,
        const_loss,
        marker="o",
        linestyle="-",
        linewidth=2,
        label=f"Constant γ={best_gamma}",
    )
    plt.plot(
        steps,
        dimi_losses,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Diminishing γₜ=1/t",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Q6: Constant vs Diminishing Learning Rate (Mini-batch SGD)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("ex2_q6_compare.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_exercise1()
    run_exercise2()
