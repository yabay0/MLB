import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

class BoundCalculator:
    """
    Compute kl and Hoeffding-based bounds for Bernoulli bias.
    """
    def __init__(self, n, delta):
        """
        Initialize sample size and confidence parameter.
        """
        self.n = n
        self.delta = delta
        self.eps = np.log(1/delta) / n

    @staticmethod
    def kl(p, q):
        """
        Compute binary kl divergence kl(p||q).
        """
        term1 = np.where(p > 0, p * np.log(p/q), 0.0)
        term2 = np.where(p < 1, (1-p) * np.log((1-p)/(1-q)), 0.0)
        return term1 + term2

    def kl_inverse_upper(self, p_hat, tol=1e-9):
        """
        Find upper inverse of kl via binary search.
        """
        lo, hi = p_hat, 1.0
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if self.kl(p_hat, mid) > self.eps:
                hi = mid
            else:
                lo = mid
        return lo

    def kl_inverse_lower(self, p_hat, tol=1e-9):
        """
        Compute lower inverse of kl by symmetry.
        """
        return 1.0 - self.kl_inverse_upper(1.0 - p_hat, tol)

    def compute_bounds(self, p_hats):
        """
        Compute Hoeffding, kl, Pinsker, and refined Pinsker bounds.
        """
        B_H = p_hats + np.sqrt(np.log(1/self.delta) / (2*self.n))
        B_KL = np.array([self.kl_inverse_upper(ph) for ph in p_hats])
        B_P = B_H.copy()
        B_RP = p_hats + np.sqrt(2*p_hats*self.eps) + 2*self.eps

        L_H = p_hats - np.sqrt(np.log(1/self.delta) / (2*self.n))
        L_KL = np.array([self.kl_inverse_lower(ph) for ph in p_hats])
        return (
            np.minimum(B_H, 1),
            np.minimum(B_KL, 1),
            np.minimum(B_P, 1),
            np.minimum(B_RP, 1),
            np.clip(L_H, 0, 1),
            np.clip(L_KL, 0, 1)
        )

class BoundPlotter:
    """
    Plot upper and lower bounds for comparison.
    """
    @staticmethod
    def plot_bounds(ax, p_hats, bounds, styles):
        """
        Plot multiple bound curves on given axes.
        """
        for bound, style in zip(bounds, styles):
            ax.plot(p_hats, bound, **style)

    @staticmethod
    def plot_full(p_hats, bounds):
        """
        Plot full-range upper bound comparison.
        """
        B_H, B_KL, B_P, B_RP, _, _ = bounds
        styles = [
            {'linestyle': '--', 'color': 'blue',   'linewidth': 2, 'label': "A. Hoeffding"},
            {'linestyle': '-',  'color': 'green',  'linewidth': 2, 'label': "B. KL bound"},
            {'linestyle': '-.', 'color': 'orange', 'linewidth': 2, 'label': "C. Pinsker"},
            {'linestyle': ':',  'color': 'red',    'linewidth': 2, 'label': "D. Refined Pinsker"}
        ]
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        BoundPlotter.plot_bounds(ax, p_hats, (B_H, B_KL, B_P, B_RP), styles)
        ax.set_xlabel("Empirical average $\\hat p_n$")
        ax.set_ylabel("Upper bound on $p$")
        ax.set_title("Comparison of four upper bounds")
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig("question_1_bounds_plot.png", dpi=300)
        plt.show()

    @staticmethod
    def plot_with_inset(p_hats, bounds, zoom_xlim=(0, 0.1)):
        """
        Plot upper bounds with zoomed inset around small p values.
        """
        B_H, B_KL, B_P, B_RP, _, _ = bounds
        styles = [
            {'linestyle': '--', 'color': 'blue',   'linewidth': 2, 'label': "A. Hoeffding"},
            {'linestyle': '-',  'color': 'green',  'linewidth': 2, 'label': "B. KL bound"},
            {'linestyle': '-.', 'color': 'orange', 'linewidth': 2, 'label': "C. Pinsker"},
            {'linestyle': ':',  'color': 'red',    'linewidth': 2, 'label': "D. Refined Pinsker"}
        ]
        fig, ax = plt.subplots(figsize=(10, 6))
        BoundPlotter.plot_bounds(ax, p_hats, (B_H, B_KL, B_P, B_RP), styles)
        ax.set_xlabel("Empirical average $\\hat p_n$")
        ax.set_ylabel("Upper bound on $p$")
        ax.set_title("Comparison of four bounds zoomed in on $[0, 0.1]$")
        ax.legend(loc='upper left')

        axins = zoomed_inset_axes(ax, zoom=6, loc='upper right',
                                  bbox_to_anchor=(0.97, 0.98),
                                  bbox_transform=ax.transAxes)
        BoundPlotter.plot_bounds(axins, p_hats, (B_H, B_KL, B_P, B_RP), styles)
        axins.set_xlim(*zoom_xlim)
        mask = (p_hats >= zoom_xlim[0]) & (p_hats <= zoom_xlim[1])
        ymin = min(bound[mask].min() for bound in (B_H, B_KL, B_P, B_RP))
        ymax = max(bound[mask].max() for bound in (B_H, B_KL, B_P, B_RP))
        axins.set_ylim(ymin, ymax)
        axins.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        plt.tight_layout()
        plt.savefig("question_1_bounds_plot_zoomed.png", dpi=300)
        plt.show()

    @staticmethod
    def plot_lower(p_hats, bounds):
        """
        Plot comparison of lower bounds.
        """
        _, _, _, _, L_H, L_KL = bounds
        plt.figure(figsize=(10, 6))
        plt.plot(p_hats, L_H, '--', linewidth=2, label="E. Hoeffding lower bound")
        plt.plot(p_hats, L_KL, '-',  linewidth=2, label="F. KL lower bound")
        plt.xlabel("Empirical average $\\hat p_n$")
        plt.ylabel("Lower bound on $p$")
        plt.title("Comparison of two lower bounds")
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig("question_1_lower_bounds.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    calculator = BoundCalculator(n=1000, delta=0.01)

    sample_ps = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    print(f"{'p_hat':>6}   {'Hoeffding':>10}   {'KL bound':>9} "
          f"{'Pinsker':>8}   {'Refined Pinsker':>15}")
    for ph in sample_ps:
        B_H  = min(ph + np.sqrt(np.log(1/calculator.delta)/(2*calculator.n)), 1.0)
        B_KL = min(calculator.kl_inverse_upper(ph), 1.0)
        B_P  = B_H
        B_RP = min(ph + np.sqrt(2*ph*calculator.eps) + 2*calculator.eps, 1.0)
        print(f"{ph:6.2f}   {B_H:10.6f}   {B_KL:9.6f}   {B_P:8.6f}   {B_RP:15.6f}")

    p_hats = np.linspace(0, 1, 200)
    bounds = calculator.compute_bounds(p_hats)
    BoundPlotter.plot_full(p_hats, bounds)
    BoundPlotter.plot_with_inset(p_hats, bounds, zoom_xlim=(0, 0.1))
    BoundPlotter.plot_lower(p_hats, bounds)
