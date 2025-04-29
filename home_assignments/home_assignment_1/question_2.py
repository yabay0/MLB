import numpy as np
import matplotlib.pyplot as plt

class BoundPlotter:
    """
    Compare classical Occam bound vs Occam's kl-razor fast-rate bound.
    """
    def __init__(self, delta=0.05, pi_h=1.0, hat_L=0.001):
        """
        Initialize parameters for bound plots.
        """
        self.delta = delta
        self.pi_h = pi_h
        self.hat_L = hat_L
        self.ln_term = np.log(1 / (pi_h * delta))

    def occam_margin(self, n):
        """
        Compute classical Occam generalization-gap bound: O(sqrt(ln(1/(πδ))/n)).
        """
        return np.sqrt(self.ln_term / n)

    def fast_margin(self, n):
        """
        Compute fast-rate kl-razor bound: O(√(2L̂ ln(1/(πδ))/n)+2 ln(1/(πδ))/n).
        """
        return np.sqrt(2 * self.hat_L * self.ln_term / n) + 2 * self.ln_term / n

    def plot(self, n_values, save_path="question_2_occam_vs_fast_rate.png"):
        """
        Plot both generalization-gap bounds over a range of n on log–log scale.
        """
        occam_vals = self.occam_margin(n_values)
        fast_vals = self.fast_margin(n_values)

        plt.figure(figsize=(10, 6))
        plt.loglog(n_values, occam_vals, label="Theorem 3.3 (Occam’s): $\\sqrt{\\ln(1/(\\pi\\delta))/n}$", linewidth=2)
        plt.loglog(n_values, fast_vals, label="Corollary 3.39 (KL-razor): $\\sqrt{2\\hat L\\ln(1/(\\pi\\delta))/n}+2\\ln(1/(\\pi\\delta))/n$", linewidth=2)
        plt.xlabel('Sample size $n$')
        plt.ylabel('Bound on $L(h)-\\hat L(h,S)$')
        plt.title(f'Comparison of Theorem 3.3 vs. Corollary 3.39 ($\\hat L$={self.hat_L}, $\\delta$={self.delta})')
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

if __name__ == "__main__":
    n = np.logspace(1, 4, 200)
    plotter = BoundPlotter(delta=0.05, pi_h=1.0, hat_L=0.001)
    plotter.plot(n)
