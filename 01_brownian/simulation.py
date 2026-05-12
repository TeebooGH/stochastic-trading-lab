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
simulate_bm(T, N, seed=None) -> (t, W)
    Simulate one trajectory of standard BM on [0, T] with N steps.

simulate_bm_with_drift(T, N, x0, mu, sigma, seed=None) -> (t, B)
    Simulate one trajectory of B_t = x0 + mu*t + sigma*W_t.

plot_trajectories(T, N, n_paths, mu=0.0, sigma=1.0, x0=0.0, seed=None)
    Plot n_paths trajectories with a mean +/- 2*std envelope.

Usage
-----
>>> t, W = simulate_bm(T=1.0, N=1000)
>>> t, B = simulate_bm_with_drift(T=1.0, N=1000, x0=0.0, mu=0.05, sigma=0.2)
"""


def simulate_bm(T: float, N: int, seed=None):
    """
    Simulates one trajectory of a standard Brownian Motion (BM) on [0, T] with N steps.

    Arguments:

        T (float): Time interval upper bound
        N (int): Number of time steps, such that [0, T] is divided in the following time grid: 0 = t_0 < t_1 < ... < t_N = T with step dt = T/N,

    Returns:
        To be defined
    """
    return None


def simulate_bm_with_drift(T: float, N: int, x0: float, mu: float, sigma: float):
    """
    Simulates one trajectory of B_t = x0 + mu*t + sigma*W_t.

    Arguments:

        T (float): Time interval upper bound
        N (int): Number of time steps, such that [0, T] is divided in the following time grid: 0 = t_0 < t_1 < ... < t_N = T with step dt = T/N,

    Returns:
        To be defined
    """
    return None


def plot_trajectories(T, N, n_paths, mu=0.0, sigma=1.0, x0=0.0, seed=None):
    """
    Plot n_paths trajectories with a mean +/- 2*std envelope.
    """
