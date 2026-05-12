# Stochastic Trading Lab

> **Status : v0 — scaffold only. No implementation yet.**
> This README is a living document. It describes the target architecture and serves as a
> construction guide throughout the 12-week build. It will be rewritten as a standard
> project README once the implementation is complete.

---

## What this project is

A modular, theory-grounded library for stochastic modelling applied to quantitative trading.
The project starts from first principles — Brownian motion, Itô calculus, stopping times,
Markov chains — and builds progressively toward a reinforcement learning agent that trades
a mean-reverting spread modelled by an Ornstein-Uhlenbeck process.

The unifying thread: every financial object in this repo (price process, signal, trading
rule, stopping criterion) has an explicit stochastic process underneath it. The code and
the mathematics are meant to be read together.

---

## Motivation and scope

Most quant GitHub portfolios fall into one of two traps: too theoretical (notebooks with
no structure), or too applied (backtests with no model). This project attempts to bridge
both — starting from a graduate-level stochastic processes course (MAT4514, Télécom
SudParis, 2026) and ending at a deployable RL trading agent.

Target roles: Quant Researcher, Quant Developer, Quant Trader.

---

## Repository structure

```
stochastic-trading-lab/
│
├── README.md
│
├── 01_brownian/
│   ├── simulation.py             # Standard BM, BM with drift and diffusion
│   ├── gbm.py                    # Geometric Brownian Motion (Black-Scholes dynamics)
│   └── reflection.py             # Reflection principle, running maximum, first passage
│
├── 02_ornstein_uhlenbeck/
│   ├── ou_process.py             # Exact simulation and Euler-Maruyama discretisation
│   ├── parameter_estimation.py   # MLE and method-of-moments calibration
│   └── mean_reversion_signals.py # Entry/exit signals derived from OU hitting times
│
├── 03_rl_spread_trading/
│   ├── environment.py            # Gymnasium environment: spread ~ OU process
│   ├── agent.py                  # Policy gradient agent (PPO or REINFORCE)
│   └── backtest.py               # Out-of-sample evaluation and performance metrics
│
├── 04_stopping_times/
│   ├── ruin_problem.py           # Gambler's ruin: Monte Carlo + martingale theory
│   └── optimal_stopping.py       # Optimal liquidation framed as a stopping problem
│
├── notebooks/
│   ├── 01_brownian_exploration.ipynb
│   ├── 02_ou_calibration.ipynb
│   ├── 03_rl_agent_demo.ipynb
│   └── 04_stopping_times.ipynb
│
└── tests/
    ├── test_brownian.py
    ├── test_ou.py
    └── test_stopping.py
```

---

## Build roadmap (12 weeks, ≤ 5h/week)

This section is the construction guide. Strike through each item as it is completed.

### Week 1 — Brownian motion foundations

- [ ] Read MAT4514 §3.1–3.2 (Definition 3.1, Remark 3.2, Theorem 3.9 Donsker)
- [ ] Implement `01_brownian/simulation.py` (standard BM, then drift + vol variant)
- [ ] Implement `01_brownian/gbm.py` (GBM fan plot, §6.1 of MAT4514)
- [ ] Implement `01_brownian/reflection.py` (Monte Carlo check of reflection principle)
- [ ] Commit with docstrings. Open `notebooks/01_brownian_exploration.ipynb`.

**Theoretical anchor:** Definition 3.1 (independent Gaussian increments), Remark 3.2
(drift and diffusion), Proposition 3.8 (symmetry and scaling), Proposition 3.21
(first passage time of BM to level $a$).

**Key result to internalize:**
$$W_t - W_s \sim \mathcal{N}(0, t-s), \quad \text{independent of } \mathcal{F}_s$$
This is the only formula you need to simulate a BM correctly.

---

### Week 2 — Ornstein-Uhlenbeck process

- [ ] Read MAT4514 §6.1 in full (OU and Black-Scholes, two pages)
- [ ] Implement `02_ornstein_uhlenbeck/ou_process.py` (Euler-Maruyama + exact scheme)
- [ ] Implement `02_ornstein_uhlenbeck/parameter_estimation.py` (MLE on simulated data)
- [ ] Open `notebooks/02_ou_calibration.ipynb`. Verify that estimated params ≈ true params.

**Theoretical anchor:** The OU process satisfies the SDE
$$dX_t = -\alpha X_t \, dt + \sigma \, dB_t, \quad X_0 = x_0$$
with explicit solution (MAT4514 §6.1):
$$X_t = x_0 e^{-\alpha t} + \sigma \int_0^t e^{-\alpha(t-s)} dB_s$$
Key properties: Gaussian, stationary distribution $\mathcal{N}(0, \sigma^2/2\alpha)$,
mean-reverting with half-life $\ln 2 / \alpha$.

---

### Week 3 — Stopping times and the ruin problem

- [ ] Read MAT4514 §1.2 (Definition 1.10, Proposition 1.18 — optional stopping theorem)
- [ ] Implement `04_stopping_times/ruin_problem.py` (TP1 Ex1: simulation + Monte Carlo)
- [ ] Begin `04_stopping_times/optimal_stopping.py` (when to liquidate a spread position)
- [ ] Open `notebooks/04_stopping_times.ipynb`.

**Theoretical anchor:** The optional stopping theorem (Proposition 1.18) guarantees that
for a martingale $(X_k)$ and bounded stopping time $T \leq S < N$:
$$\mathbb{E}[X_S \mid \mathcal{F}_T] = X_T$$
Applied to the gambler's ruin: if $X_k$ is a symmetric random walk and $T = \inf\{k : X_k
\notin (-a, b)\}$, then $\mathbb{E}[X_T] = \mathbb{E}[X_0] = 0$, which gives
$P(X_T = b) = a/(a+b)$.

**Connection to trading:** the ruin problem is a stylised model for drawdown management.
The optimal stopping problem in `optimal_stopping.py` asks: given that the spread follows
an OU process, at what level should you exit to maximise expected P&L? This directly
feeds into the RL environment design.

---

### Week 4 — RL environment integration

- [ ] Read MAT4514 §5.1–5.2 (Itô process, Itô formula — enough to understand why
      Euler-Maruyama works)
- [ ] Audit `03_rl_spread_trading/environment.py`: verify the OU discretisation is correct
- [ ] Add a "Theoretical background" section to the module docstring of `environment.py`
      citing the relevant SDE and its parameters
- [ ] Write `tests/test_brownian.py` and `tests/test_ou.py` (basic distributional checks)

---

### Weeks 5–8 — Signals and calibration on real data (Direction 1)

- [ ] Fetch real spread data via `yfinance` (e.g. a cointegrated equity pair)
- [ ] Calibrate OU parameters on the real spread in `parameter_estimation.py`
- [ ] Implement `02_ornstein_uhlenbeck/mean_reversion_signals.py`: entry/exit thresholds
      derived from the first passage time $T_a$ (Proposition 3.21)
- [ ] Integrate the signal as an observation in the RL environment
- [ ] Update `notebooks/03_rl_agent_demo.ipynb` with real-data results

---

### Weeks 9–12 — Finition and publication

- [ ] Rewrite this README as a standard project README (remove the roadmap section,
      add badges, results, and a concise "how to run" guide)
- [ ] Add a "Research note" PDF (one page): problem statement, model, key results,
      performance metrics — formatted as a quant research note
- [ ] Final pass on all docstrings for consistency
- [ ] Tag `v1.0`

---

## Theoretical foundations

All theoretical results used in this project come from:

> Franceschi, S., Pieczynski, W., Schechtman, S., Flin, J.
> _Processus Stochastiques_, MAT4514, Télécom SudParis / Institut Polytechnique de Paris, 2026.

Key results by module:

| Module                      | MAT4514 reference                   | Result used                                                 |
| --------------------------- | ----------------------------------- | ----------------------------------------------------------- |
| `simulation.py`             | Definition 3.1, Remark 3.2          | BM as process with independent Gaussian increments          |
| `gbm.py`                    | §6.1 (Black-Scholes SDE)            | Itô formula applied to $f(t, B_t) = e^{\mu t + \sigma B_t}$ |
| `reflection.py`             | Proposition 3.21, Exercise II.5     | $P(\sup_{s \leq t} W_s \geq a) = 2P(W_t \geq a)$            |
| `ou_process.py`             | §6.1 (OU SDE), §4.3 (Itô integral)  | Explicit solution of the OU SDE                             |
| `parameter_estimation.py`   | §6.1 (stationary distribution)      | $X_\infty \sim \mathcal{N}(0, \sigma^2 / 2\alpha)$          |
| `mean_reversion_signals.py` | Proposition 3.21                    | Laplace transform of first passage time                     |
| `ruin_problem.py`           | Definition 1.10, Proposition 1.18   | Optional stopping theorem                                   |
| `optimal_stopping.py`       | §1.2, §1.5 (martingale convergence) | Value function as harmonic function                         |
| `environment.py`            | §6.1, §4.3                          | Euler-Maruyama discretisation of OU SDE                     |

---

## Dependencies

```
numpy
scipy
matplotlib
gymnasium
yfinance         # weeks 5–8 only
stable-baselines3  # or custom RL implementation
```

---

## How to run (placeholder — will be updated at v1.0)

```bash
git clone https://github.com/TeebooGH/stochastic-trading-lab
cd stochastic-trading-lab
pip install -r requirements.txt # This line will be replaced since the project is bootstrapped using the uv package manager.
python 01_brownian/simulation.py
```

---

## About the author

OU Thibaud, Engineering student @Télécom SudParis (DANI VAP track, 2026).

Interests: stochastic modelling, quantitative finance,
reinforcement learning.
