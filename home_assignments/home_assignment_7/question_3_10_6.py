import numpy as np
import matplotlib.pyplot as plt
from math import ceil, log2, sqrt, log

class KLUBComparison:
    """
    Numerically compare standard KL and Unexpected-Bernstein bounds for
    ternary distributions Z ∈ {0,1/2,1} using empirical variance.
    """
    def __init__(self, n=100, delta=0.05, grid_points=50):
        self.n     = n
        self.delta = delta
        self.epsilon_kl = np.log(1/self.delta) / self.n
        self.grid = np.linspace(0, 1, grid_points)
        arg = sqrt(self.n / log(1/self.delta)) / 2
        self.k = max(1, int(ceil(log2(arg))))
        self.lambdas = np.array([1 / (2**i) for i in range(1, self.k+1)])

    @staticmethod
    def kl_div(p, q):
        eps = 1e-12
        p = np.clip(p, eps, 1-eps)
        q = np.clip(q, eps, 1-eps)
        return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))

    def kl_inv_upper(self, phat, eps, tol=1e-9):
        low, high = phat, 1.0
        while high - low > tol:
            mid = 0.5*(low + high)
            if self.kl_div(phat, mid) > eps:
                high = mid
            else:
                low = mid
        return low

    def unexpected_bernstein(self, vhat):
        const = np.log(self.k / self.delta) / self.n
        terms = self.lambdas * vhat + const / self.lambdas
        return np.min(terms)

    def simulate_bounds(self):
        kl_bounds = []
        ub_bounds = []

        for p12 in self.grid:
            p0 = p1 = (1 - p12) / 2
            X = np.random.choice([0, 0.5, 1], size=self.n, p=[p0, p12, p1])

            phat = X.mean()
            vhat = ((X - phat)**2).mean()

            inv_kl = self.kl_inv_upper(phat, self.epsilon_kl)
            kl_bounds.append(inv_kl - phat)

            ub_bounds.append(self.unexpected_bernstein(vhat))

        return kl_bounds, ub_bounds

    def plot_comparison(self, kl_bounds, ub_bounds):
        plt.figure(figsize=(10, 6))
        plt.plot(self.grid, kl_bounds, label='kl bound')
        plt.plot(self.grid, ub_bounds, label='Unexpected-Bernstein bound')
        plt.xlabel('$p_{1/2}=P(Z=1/2)$')
        plt.ylabel('Bound on $p - \hat p_n$')
        plt.title("kl vs Unexpected-Bernstein Bound")
        plt.legend()
        plt.tight_layout()
        plt.savefig("kl_vs_unexpected_bernstein.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    comp = KLUBComparison(n=100, delta=0.05, grid_points=200)
    kl_b, ub_b = comp.simulate_bounds()
    comp.plot_comparison(kl_b, ub_b)