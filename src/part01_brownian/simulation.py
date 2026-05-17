# =============================================================================
# 01_brownian/simulation.py
#
# Author : Thibaud OU
# Date : May 13th, 2026
# =============================================================================

"""
Standard Brownian Motion — Simulation and Visualisation
========================================================

Implements simulation of:
  - Standard Brownian motion  W_t
  - Brownian motion with drift and diffusion  B_t = x + mu*t + sigma*W_t

Theoretical foundation
----------------------
A standard Brownian motion (W_t)_{t >= 0} is a stochastic process satisfying
(MAT4514, Definition 3.1):

  (i)   Continuous trajectories almost surely.
  (ii)  Independent increments: for 0 <= t_1 < ... < t_n, the increments
        W_{t_2} - W_{t_1}, ..., W_{t_n} - W_{t_{n-1}} are independent.
  (iii) Stationary increments: W_t - W_s ~ N(0, t-s) for all s <= t.
  (iv)  W_0 = 0.

Simulation principle (Euler-Maruyama at dt resolution):
  Given a time grid 0 = t_0 < t_1 < ... < t_N = T with step dt = T/N,
  simulate increments

      dW_k = W_{t_{k+1}} - W_{t_k} ~ N(0, dt)   i.i.d.

  and set W_{t_{k+1}} = W_{t_k} + dW_k.

  This is exact for standard BM (no approximation error) because BM has
  Gaussian increments by definition.

For a BM with drift mu and diffusion sigma starting at x (Remark 3.2):
      B_t = x + mu*t + sigma*W_t
  so  dB_k = mu*dt + sigma*sqrt(dt)*Z_k,  Z_k ~ N(0,1) i.i.d.

Connection to Donsker's theorem (Theorem 3.9, MAT4514):
  The rescaled symmetric random walk S^(n)_t = S_{floor(nt)} / sqrt(n)
  converges in law to standard BM. This module is the continuous-time limit
  of that discrete construction.

Functions
---------
simulate_bm_matrix(T, N, n_paths: int | None = None, seed=None) -> (t, W)
    Simulate n_paths trajectories of standard BM on [0, T] with N steps.

simulate_drifted_bm_matrix(T, N, x0, mu, sigma, n_paths: int | None = None, seed=None) -> (t, B)
    Simulate n_paths trajectories of B_t = x0 + mu*t + sigma*W_t.

plot_trajectories(T, N, n_paths, mu=0.0, sigma=1.0, x0=0.0, seed=None)
    Plot n_paths trajectories with a mean +/- 2*std envelope.

Usage
-----
>>> t, W = simulate_bm_matrix(T=1.0, N=1000, n_paths=10000)
>>> t, B = simulate_drifted_bm_matrix(T=1.0, N=1000, n_paths=10000, x0=0.0, mu=0.05, sigma=0.2)
"""

import matplotlib.pyplot as plt
import numpy as np


def simulate_bm_matrix(
    T: float, N: int, n_paths: int | None = None, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates multiple trajectories of a Standard Brownian Motion (SBM) simultaneously.

    Arguments:
        T (float): Time interval upper bound.
        N (int): Number of time steps.
        n_paths (int | None): Number of independent trajectories.
                              If None, returns a 1D array (single path).
                              If int, returns a 2D array of shape (n_paths, N+1).
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - t (np.ndarray): Time grid array of shape (N+1,).
            - W (np.ndarray): Matrix of BM trajectories of shape (n_paths, N+1).
                              Each row is an independent path starting at W_0 = 0.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    t = np.linspace(0, T, N + 1)

    # Define the size of the Gaussian increments vector to handle 1D and 2D cases
    size = N if n_paths is None else (n_paths, N)

    gaussian_increments = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=size)

    # axis=-1 handles the 1D and 2D cases simultaneously
    W_increments = np.cumsum(gaussian_increments, axis=-1)

    if n_paths is None:
        W = np.insert(W_increments, 0, 0.0)
    else:
        initial_conditions = np.zeros((n_paths, 1))
        W = np.hstack((initial_conditions, W_increments))

    return t, W


def simulate_drifted_bm_matrix(
    T: float,
    N: int,
    x0: float,
    mu: float,
    sigma: float,
    n_paths: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates trajectories of a Brownian Motion with drift: B_t = x0 + mu*t + sigma*W_t.

    Arguments:
        T (float): Time interval upper bound
        N (int): Number of time steps.
        x0 (float): Initial value.
        mu (float): Drift coefficient.
        sigma (float): Diffusion coefficient (volatility).
        n_paths (int | None): Number of independent trajectories.
                              If None, returns a 1D array (single path).
                              If int, returns a 2D array of shape (n_paths, N+1).
        seed (int | None): Random seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - t (np.ndarray): Time grid array of shape (N+1, ).
            - B (np.ndarray): BM trajectory array. Shape is (N+1,) if n_paths is None,
                              else (n_paths, N+1).
    """
    t, W = simulate_bm_matrix(T, N, n_paths, seed)
    B = x0 + mu * t + sigma * W

    return t, B


def plot_trajectories(
    T: float, N: int, n_paths: int, mu=0.0, sigma=1.0, x0=0.0, seed=None
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plots n_paths trajectories with a mean +/- 2*std empirical and theoretical envelope.

    Arguments:
        T (float): Time interval upper bound.
        N (int): Number of time steps.
        n_paths (int): Number of independent trajectories to simulate.
        mu (float): Drift coefficient. Defaults to 0.0.
        sigma (float): Diffusion coefficient. Defaults to 1.0.
        x0 (float): Initial value. Defaults to 0.0.
        seed (int | None): Random seed. Defaults to None.

    Returns:
        tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]: The figure and a tuple containing the two subplots.
    """

    t, B = simulate_drifted_bm_matrix(T, N, x0, mu, sigma, n_paths=n_paths, seed=seed)

    # Evaluate empirical estimators, ddof = 1 for std in order to evaluate an unbiased estimator
    emp_mean = np.mean(B, axis=0)
    emp_std = np.std(B, axis=0, ddof=1)

    # Theoretical values
    theory_mean = x0 + mu * t
    theory_std = sigma * np.sqrt(t)

    # Visualization using matplotlib.pyplot
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # ---------------------------------------------------------
    # Subplot 1 : Trajectories & Theoretical Bounds
    # ---------------------------------------------------------
    # Only drawing 100 trajectories in order not to saturate the plot
    ax1.plot(t, B[:100].T, color="slategray", alpha=0.15, linewidth=0.5)

    # Mean line plots
    ax1.plot(
        t,
        theory_mean,
        color="firebrick",
        linestyle="--",
        linewidth=2,
        label=r"Theoretical Mean $\mathbb{E}[B_t]$",
    )
    ax1.plot(
        t,
        emp_mean,
        color="navy",
        linewidth=1.5,
        label=r"Empirical Mean $\hat{\mathbb{E}}[B_t]$",
    )

    # Theoretical bounds (variance signal)
    theory_y_inf = theory_mean - 2 * theory_std
    theory_y_sup = theory_mean + 2 * theory_std
    ax1.fill_between(
        t,
        theory_y_inf,
        theory_y_sup,
        color="firebrick",
        alpha=0.1,
        label=r"Theoretical 95% CI ($\pm 2\sigma\sqrt{t}$)",
    )

    ax1.set_xlabel(r"Time $t$ (Years)")
    ax1.set_ylabel(r"Process Value $B_t$")
    ax1.set_title(
        f"Drifted Brownian Motion Dynamics ($x_0={x0}, \\mu={mu}, \\sigma={sigma}$)"
    )
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Subplot 2 : Convergence Analysis (Limits)
    # ---------------------------------------------------------
    emp_y_inf = emp_mean - 2 * emp_std
    emp_y_sup = emp_mean + 2 * emp_std

    # Color coding : Red for Theory, Blue for Empirical
    ax2.plot(t, theory_y_sup, ls="--", color="firebrick", label=r"Theory Upper Bound")
    ax2.plot(t, theory_y_inf, ls="--", color="firebrick", label=r"Theory Lower Bound")

    ax2.plot(
        t,
        emp_y_sup,
        color="navy",
        linewidth=1.5,
        alpha=0.8,
        label=r"Empirical Upper Bound",
    )
    ax2.plot(
        t,
        emp_y_inf,
        color="navy",
        linewidth=1.5,
        alpha=0.8,
        label=r"Empirical Lower Bound",
    )

    # Filling the difference between Emp and Theo to visualize the convergence errors
    ax2.fill_between(t, theory_y_sup, emp_y_sup, color="gray", alpha=0.3)
    ax2.fill_between(t, theory_y_inf, emp_y_inf, color="gray", alpha=0.3)

    ax2.set_xlabel(r"Time $t$ (Years)")
    ax2.set_ylabel(r"Process Value $B_t$")
    ax2.set_title(f"Convergence of 95% Confidence Intervals ($K={n_paths}$ paths)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig, (ax1, ax2)
