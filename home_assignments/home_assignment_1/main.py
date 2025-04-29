import numpy as np

from question_1 import BoundCalculator, BoundPlotter as Q1Plotter
from question_2 import BoundPlotter as OccamPlotter
from question_3 import KLSplitComparison


def run_question_1():
    """
    Question 1: Compare four upper and two lower bounds on Bernoulli bias p.
    Major steps:
      1. Initialize BoundCalculator with sample size n and confidence δ.
      2. Print numerical table of bounds at specific p̂ values.
      3. Generate a fine grid of p̂ across [0,1] and compute all bounds.
      4. Plot:
         - Full-range upper bounds (Hoeffding, kl, Pinsker, Refined Pinsker).
         - Zoomed-in inset on [0,0.1] to highlight fast rates near zero.
         - Lower bounds comparison (Hoeffding vs. kl lower).
    """
    n, delta = 1000, 0.01
    calculator = BoundCalculator(n=n, delta=delta)

    sample_ps = np.arange(0, 0.12, 0.02)
    print(f"{'p_hat':>6}   {'Hoeffding':>10}   {'kl bound':>9}   {'Pinsker':>8}   {'Refined Pinsker':>15}")
    for ph in sample_ps:
        B_H, B_KL, B_P, B_RP, _, _ = calculator.compute_bounds(np.array([ph]))
        print(f"{ph:6.2f}   {B_H[0]:10.6f}   {B_KL[0]:9.6f}   {B_P[0]:8.6f}   {B_RP[0]:15.6f}")

    p_hats = np.linspace(0, 1, 200)
    bounds = calculator.compute_bounds(p_hats)

    Q1Plotter.plot_full(p_hats, bounds)
    Q1Plotter.plot_with_inset(p_hats, bounds, zoom_xlim=(0, 0.1))
    Q1Plotter.plot_lower(p_hats, bounds)


def run_question_2():
    """
    Question 2: Occam's kl-razor vs classical Occam (fast vs slow rates).
    Major steps:
      1. Define a range of sample sizes n (from 10^1 to 10^4).
      2. Initialize OccamPlotter with δ, prior weight π(h), and fixed empirical loss L̂.
      3. Compute and plot both generalization-gap bounds on a log–log scale.
    """
    n_values = np.logspace(1, 4, 200)
    plotter = OccamPlotter(delta=0.05, pi_h=1.0, hat_L=0.001)
    plotter.plot(n_values)


def run_question_3():
    """
    Question 3: Compare standard KL vs split-KL bounds for ternary distributions.
    Major steps:
      1. Initialize KLSplitComparison with n, δ, and grid resolution.
      2. Over grid of component weight p_{1/2}, simulate n samples of {0,1/2,1}.
      3. Compute:
         - Standard kl bound on p - p̂_n via upper-inverse kl.
         - Split-kl bound by decomposing into two Bernoulli events.
      4. Plot both bounds as functions of p_{1/2}.
    """
    comp = KLSplitComparison(n=100, delta=0.05, grid_points=200)
    kl_bounds, split_bounds = comp.simulate_bounds()
    comp.plot_comparison(kl_bounds, split_bounds)


def main():
    run_question_1()
    run_question_2()
    run_question_3()


if __name__ == "__main__":
    main()