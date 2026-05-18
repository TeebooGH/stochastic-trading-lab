# =============================================================================
# main1.py
#
# Minimalist CLI orchestrator for the Stochastic Trading Lab.
# =============================================================================

import sys
import threading
import time

import matplotlib.pyplot as plt

from part01_brownian.gbm import simulate_gbm

# Importing the core engine modules
from part01_brownian.simulation import (
    plot_trajectories,
    simulate_bm_matrix,
    simulate_drifted_bm_matrix,
)


# --- Async UI Loader ---
class AsymptoticLoader:
    def __init__(self, message, style="ascii"):
        self.message = message
        self.style = style
        self.is_running = False
        self.thread = None
        self.progress = 0
        self.max_steps = 20

    def start(self):
        self.is_running = True
        self.progress = 0
        self.thread = threading.Thread(target=self._animate)
        self.thread.start()

    def _animate(self):
        delay = 0.05
        while self.is_running and self.progress < self.max_steps - 1:
            self._draw(self.progress)
            time.sleep(delay)
            self.progress += 1
            delay *= 1.25

        while self.is_running:
            self._draw(self.progress)
            time.sleep(0.1)

    def _draw(self, step):
        bar = "=" * step + " " * (self.max_steps - step)
        sys.stdout.write(f"\r  Status: {self.message} [{bar}]")
        sys.stdout.flush()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        self._draw(self.max_steps)
        time.sleep(0.1)
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()


def run_with_loader(func, message, style, *args, **kwargs):
    print()
    loader = AsymptoticLoader(message, style)
    loader.start()
    try:
        result = func(*args, **kwargs)
    finally:
        loader.stop()
    return result


# --- Input Helper Functions ---
def prompt_float(message: str, default: float) -> float:
    val = input(f"  [?] {message} [{default}]: ").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"  [!] Invalid input, using default: {default}")
        return default


def prompt_int(message: str, default: int) -> int:
    val = input(f"  [?] {message} [{default}]: ").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"  [!] Invalid input, using default: {default}")
        return default


# --- Post-Simulation Sub-Menu ---
def post_simulation_menu(t, paths, model_name, params_dict=None):
    while True:
        print(f"\n--- [ ANALYSIS OPTIONS: {model_name} ] ---")
        print("  [1] Plot Sample Trajectories")
        print("  [2] Plot Convergence Analysis (Fan Plot)")
        print("  [0] Return to Main Menu")

        choice = input("  > Select analysis (0-2): ").strip()
        if choice == "0":
            break
        elif choice == "1":
            _plot_sample(t, paths, model_name)
        elif choice == "2":
            _plot_convergence(t, paths, model_name, params_dict)
        else:
            print("  [!] Invalid choice. Please select 0, 1, or 2.")


def _plot_sample(t, paths, model_name):
    plt.figure(figsize=(10, 5))
    if paths.ndim > 1:
        max_paths = min(50, paths.shape[0])
        plt.plot(t, paths[:max_paths].T, alpha=0.3, linewidth=1.0)
        plt.title(f"{model_name} (Sample of {max_paths} Paths)")
    else:
        plt.plot(t, paths, color="#2c3e50", linewidth=1.5)
        plt.title(f"{model_name} (Single Path)")
    plt.xlabel("Time (t)")
    plt.ylabel("Process Value")
    plt.grid(True, alpha=0.3)
    plt.show()


def _plot_convergence(t, paths, model_name, p):
    if p is None:
        print(
            "\n  [!] Convergence bounds are not mathematically defined in this module for this process."
        )
        return
    if paths.ndim == 1 or paths.shape[0] < 10:
        print(
            "\n  [!] Need at least 10 paths for a meaningful Monte Carlo convergence analysis."
        )
        return

    fig, axes = run_with_loader(
        plot_trajectories,
        "Generating Convergence Analysis...",
        "ascii",
        T=t[-1],
        N=len(t) - 1,
        n_paths=paths.shape[0],
        mu=p.get("mu", 0.0),
        sigma=p.get("sigma", 1.0),
        x0=p.get("x0", 0.0),
    )
    plt.show()


# --- Simulation Runners ---
def run_bm():
    print("\n--- [ MODULE: Standard Brownian Motion ] ---")
    T = prompt_float("Total time T (years)", 1.0)
    N = prompt_int("Number of steps N", 1000)
    n_paths = prompt_int("Number of paths to simulate", 500)

    t, W = run_with_loader(
        simulate_bm_matrix,
        "Computing trajectories...",
        "ascii",
        T=T,
        N=N,
        n_paths=n_paths,
    )
    print(f"  Status: Simulation complete ({n_paths} paths, {N} steps).")

    params = {"mu": 0.0, "sigma": 1.0, "x0": 0.0}
    post_simulation_menu(t, W, "Standard Brownian Motion", params)


def run_drifted_bm():
    print("\n--- [ MODULE: Drifted Brownian Motion ] ---")
    T = prompt_float("Total time T (years)", 1.0)
    N = prompt_int("Number of steps N", 1000)
    n_paths = prompt_int("Number of paths to simulate", 500)
    x0 = prompt_float("Initial value x0", 0.0)
    mu = prompt_float("Drift (mu)", 0.05)
    sigma = prompt_float("Volatility (sigma)", 0.2)

    t, B = run_with_loader(
        simulate_drifted_bm_matrix,
        "Computing trajectories...",
        "ascii",
        T=T,
        N=N,
        x0=x0,
        mu=mu,
        sigma=sigma,
        n_paths=n_paths,
    )
    print(f"  Status: Simulation complete ({n_paths} paths, {N} steps).")

    params = {"mu": mu, "sigma": sigma, "x0": x0}
    post_simulation_menu(t, B, "Drifted Brownian Motion", params)


def run_gbm():
    print("\n--- [ MODULE: Geometric Brownian Motion ] ---")
    T = prompt_float("Total time T (years)", 1.0)
    N = prompt_int("Number of steps N", 1000)
    n_paths = prompt_int("Number of paths to simulate", 500)
    s0 = prompt_float("Initial price S0", 100.0)
    mu = prompt_float("Expected return (mu)", 0.08)
    sigma = prompt_float("Volatility (sigma)", 0.2)

    t, S = run_with_loader(
        simulate_gbm,
        "Computing trajectories...",
        "ascii",
        T=T,
        N=N,
        s0=s0,
        mu=mu,
        sigma=sigma,
        n_paths=n_paths,
    )
    print(f"  Status: Simulation complete ({n_paths} paths, {N} steps).")

    post_simulation_menu(t, S, "Geometric Brownian Motion", params_dict=None)


def main():
    while True:
        print(
            "\n+-----------------------------------------------------------------------+"
        )
        print(
            "|                  STOCHASTIC TRADING LAB - CLI                         |"
        )
        print(
            "|              A personal project by Thibaud Ou                         |"
        )
        print(
            "+-----------------------------------------------------------------------+"
        )
        print("  [1] Simulate Standard Brownian Motion")
        print("  [2] Simulate Drifted Brownian Motion")
        print("  [3] Simulate Geometric Brownian Motion (GBM)")
        print("  [4] Simulate Ornstein-Uhlenbeck Process (OU)")
        print("  [5] Ruin Problem & Stopping Times")
        print("  [6] Run RL Agent (Spread Trading)")
        print("  [0] Exit")
        print(
            "-------------------------------------------------------------------------"
        )

        choice = input("  > Select an option (0-6): ").strip()

        if choice == "1":
            run_bm()
        elif choice == "2":
            run_drifted_bm()
        elif choice == "3":
            run_gbm()
        elif choice in ["4", "5", "6"]:
            print("\n  [!] Notice: This module is currently under development.")
        elif choice == "0":
            print("\n  Exiting. Goodbye.\n")
            sys.exit(0)
        else:
            print("\n  [!] Error: Invalid choice. Please input a valid module number.")


if __name__ == "__main__":
    main()
