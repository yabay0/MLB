import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def kl_div(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def invert_kl(p, rhs, tol=1e-6, max_iter=50):
    low, high = p, 1 - tol
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if kl_div(p, mid) <= rhs:
            low = mid
        else:
            high = mid
        if high - low < tol:
            break
    return low


# ----------------------------------------------------------------------
# Data loading & preprocessing
# ----------------------------------------------------------------------
def load_ionosphere(path="ionosphere.data", seed=42):
    cols = [f"f{i+1}" for i in range(34)] + ["label"]
    df = pd.read_csv(path, header=None, names=cols)
    df["label"] = df["label"].map({"g": +1, "b": -1})
    X = df.iloc[:, :-1].values
    y = df["label"].values
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=200, test_size=150, stratify=y, random_state=seed
    )
    return X_tr, X_te, y_tr, y_te


# ----------------------------------------------------------------------
# Baseline: CV‐tuned SVM
# ----------------------------------------------------------------------
def median_gamma(X, y):
    n = X.shape[0]
    d2 = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
    mask = y[:, None] != y[None, :]
    mins = np.min(np.where(mask, d2, np.inf), axis=1)
    return 1.0 / (2 * np.median(mins))


def baseline_cv_svm(X_tr, y_tr, X_te, y_te, reps=10, seed=0):
    gamma0 = median_gamma(X_tr, y_tr)
    gamma_grid = gamma0 * np.logspace(-4, 4, 9)
    C_grid = np.logspace(-3, 3, 7)

    mean_losses = []
    mean_times = []

    for _ in range(reps):
        svc = SVC(kernel="rbf")
        param_grid = {"C": C_grid, "gamma": gamma_grid}
        grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)

        t0 = time.perf_counter()
        grid.fit(X_tr, y_tr)
        clf = grid.best_estimator_
        clf.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - t0

        loss = np.mean(clf.predict(X_te) != y_te)
        mean_losses.append(loss)
        mean_times.append(elapsed)

    return (
        np.mean(mean_losses),
        np.std(mean_losses),
        np.mean(mean_times),
        np.std(mean_times),
        grid.best_params_["C"],
        grid.best_params_["gamma"],
    )


# ----------------------------------------------------------------------
# PAC‐Bayes aggregation
# ----------------------------------------------------------------------
def pacbayes_aggregation(
    X_tr, y_tr, X_te, y_te, ms, r, delta, gamma_grid, C, reps=10, seed=0
):
    n = len(y_tr)
    mean_losses, std_losses = [], []
    mean_times, std_times = [], []
    mean_bounds, std_bounds = [], []
    rng = np.random.RandomState(seed)

    for m in ms:
        losses, times, bounds = [], [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            # Draw m subsets of r samples
            subsets = [rng.choice(n, r, replace=False) for _ in range(m)]
            # Compute validation losses and test predictions
            val_losses = np.zeros(m)
            test_preds = np.zeros((m, len(y_te)))
            for i, idx in enumerate(subsets):
                gamma = rng.choice(gamma_grid)
                clf = SVC(kernel="rbf", C=C, gamma=gamma)
                clf.fit(X_tr[idx], y_tr[idx])
                mask = np.ones(n, bool)
                mask[idx] = False
                val_losses[i] = np.mean(clf.predict(X_tr[mask]) != y_tr[mask])
                test_preds[i] = clf.predict(X_te)
            # Alternating minimization of PAC-Bayes-λ
            lam = 1.0
            Lmin = val_losses.min()
            for _ in range(200):
                w = np.exp(-lam * (n - r) * (val_losses - Lmin))
                rho = w / w.sum()
                KL = np.sum(rho * np.log(rho * m))
                E_L = rho.dot(val_losses)
                lam_new = 2 / (
                    np.sqrt(
                        2 * (n - r) * E_L / (KL + np.log(2 * np.sqrt(n - r) / delta))
                        + 1
                    )
                    + 1
                )
                if abs(lam_new - lam) < 1e-6:
                    lam = lam_new
                    break
                lam = lam_new
            # Compute rho-weighted vote loss and bound
            y_pred = np.sign(rho @ test_preds)
            losses.append(np.mean(y_pred != y_te))
            rhs = (KL + np.log(2 * np.sqrt(n - r) / delta)) / (n - r)
            bounds.append(invert_kl(rho.dot(val_losses), rhs))
            times.append(time.perf_counter() - t0)

        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))
        mean_times.append(np.mean(times))
        std_times.append(np.std(times))
        mean_bounds.append(np.mean(bounds))
        std_bounds.append(np.std(bounds))

    return (
        np.array(mean_losses),
        np.array(std_losses),
        np.array(mean_times),
        np.array(std_times),
        np.array(mean_bounds),
        np.array(std_bounds),
    )


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------
def plot_no_std(ms, L_cv, L_m, q_m, t_cv, t_m):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    ax1.plot(ms, L_m, "k-", label="Our Method")
    ax1.plot(ms, [L_cv] * len(ms), "r-", label="CV SVM")
    ax1.plot(ms, q_m, "b-", label="Bound")
    ax2.plot(ms, t_m, "k--", label=r"$t_m$")
    ax2.plot(ms, [t_cv] * len(ms), "r--", label=r"$t_{cv}$")
    ax1.set_xscale("log")
    ax1.set_xlabel("m")
    ax1.set_ylabel("Test loss")
    ax2.set_ylabel("Runtime (s)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(h1 + h2, l1 + l2, loc="upper right", bbox_to_anchor=(1, 0.90), frameon=True)
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    plt.title(
        "PAC‐Bayesian aggregation vs. RBF kernel SVM tuned by cross‐validation\n(mean of 10 reps)",
        fontsize=10,
    )
    plt.savefig("pac_bayes_vs_svm_mean.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_with_std(ms, L_cv, L_m, L_std, q_m, q_std, t_cv, t_m, t_std):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    ax1.errorbar(ms, L_m, yerr=L_std, fmt="k-", capsize=3, label="Our Method ± std")
    ax1.plot(ms, [L_cv] * len(ms), "r-", label="CV SVM")
    ax1.plot(ms, q_m, "b-", label="Bound")
    ax1.fill_between(ms, q_m - q_std, q_m + q_std, color="tab:blue", alpha=0.2)
    ax2.errorbar(ms, t_m, yerr=t_std, fmt="k--", capsize=3, label=r"$t_m$ ± std")
    ax2.plot(ms, [t_cv] * len(ms), "r--", label=r"$t_{cv}$")
    ax1.set_xscale("log")
    ax1.set_xlabel("m")
    ax1.set_ylabel("Test loss")
    ax2.set_ylabel("Runtime (s)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Move legend a bit down vertically
    leg = ax1.legend(h1 + h2, l1 + l2, loc="upper right", bbox_to_anchor=(1, 0.90), frameon=True)
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    plt.title(
        "PAC‐Bayesian aggregation vs. RBF kernel SVM tuned by cross‐validation\n(mean ± std of 10 reps)",
        fontsize=10,
    )
    plt.savefig("pac_bayes_vs_svm_mean_std.png", dpi=300, bbox_inches="tight")
    plt.show()


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------
def main():
    # load
    X_tr, X_te, y_tr, y_te = load_ionosphere()
    n, d = X_tr.shape
    r = d + 1
    delta = 0.05

    # baseline
    L_cv, Lcv_std, t_cv, tcv_std, best_C, best_gamma = baseline_cv_svm(
        X_tr, y_tr, X_te, y_te, reps=10, seed=42
    )

    # pac-bayes
    ms = np.unique(np.round(np.logspace(0, np.log10(n), 20))).astype(int)
    gamma0 = median_gamma(X_tr, y_tr)
    gamma_grid = gamma0 * np.logspace(-4, 4, 9)
    L_m, L_std, t_m, t_std, q_m, q_std = pacbayes_aggregation(
        X_tr, y_tr, X_te, y_te, ms, r, delta, gamma_grid, best_C, reps=10, seed=42
    )

    # plots
    plot_no_std(ms, L_cv, L_m, q_m, t_cv, t_m)
    plot_with_std(ms, L_cv, L_m, L_std, q_m, q_std, t_cv, t_m, t_std)


if __name__ == "__main__":
    main()