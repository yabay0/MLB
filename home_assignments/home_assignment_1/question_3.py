import numpy as np
import matplotlib.pyplot as plt

class KLSplitComparison:
    """
    Numerically compare standard kl and split-kl bounds for ternary distributions.
    """
    def __init__(self, n=100, delta=0.05, grid_points=50):
        """
        Initialize parameters and grid for comparison.
        """
        self.n = n
        self.delta = delta
        self.epsilon_kl    = np.log(1/self.delta)       / self.n
        self.epsilon_split = np.log(2/self.delta)       / self.n
        self.grid          = np.linspace(0, 1, grid_points)

    @staticmethod
    def kl_div(p, q):
        """
        Compute binary kl divergence kl(p||q) with clipping for numerical stability.
        """
        eps = 1e-12
        p = np.clip(p, eps, 1-eps)
        q = np.clip(q, eps, 1-eps)
        return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))

    def kl_inv_upper(self, phat, eps, tol=1e-9):
        """
        Find the maximum q >= phat such that kl(phat||q) <= eps via binary search.
        """
        low, high = phat, 1.0
        while high - low > tol:
            mid = 0.5 * (low + high)
            if self.kl_div(phat, mid) > eps:
                high = mid
            else:
                low = mid
        return low

    def simulate_bounds(self):
        """
        Simulate kl and split-kl bound differences over the p_{1/2} grid.
        """
        kl_bounds = []
        split_bounds = []
        for p12 in self.grid:
            p0 = (1 - p12) / 2
            p1 = p0
            X = np.random.choice([0, 0.5, 1], size=self.n, p=[p0, p12, p1])
            phat = X.mean()
            phat1 = (X >= 0.5).mean()
            phat2 = (X >= 1.0).mean()

            inv_kl = self.kl_inv_upper(phat, self.epsilon_kl)
            kl_bounds.append(inv_kl - phat)

            inv1 = self.kl_inv_upper(phat1, self.epsilon_split)
            inv2 = self.kl_inv_upper(phat2, self.epsilon_split)
            split_bounds.append(0.5 * inv1 + 0.5 * inv2 - phat)

        return kl_bounds, split_bounds

    def plot_comparison(self, kl_bounds, split_bounds):
        """
        Plot comparison of kl vs split-kl bound on p - \hat p over the p_{1/2} grid.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.grid, kl_bounds, label='kl bound')
        plt.plot(self.grid, split_bounds, label='Split-kl bound')
        plt.xlabel('$p_{1/2}=P(X=1/2)$')
        plt.ylabel('Bound on $p-\hat p_n$')
        plt.title(f'Comparison of kl and split-kl bounds (n={self.n}, δ={self.delta})')
        plt.legend()
        plt.tight_layout()
        plt.savefig("question_3_split_kl_bound.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    comp = KLSplitComparison(n=100, delta=0.05, grid_points=200)
    kl_b, split_b = comp.simulate_bounds()
    comp.plot_comparison(kl_b, split_b)