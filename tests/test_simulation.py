import numpy as np
import pytest

from part01_brownian.simulation import simulate_bm_matrix, simulate_drifted_bm_matrix

T = 1.0
N = 100
x0 = 0.0
mu = 0.0
sigma = 1.0


@pytest.mark.parametrize(
    "n_paths_arg, expected_shape",
    [
        (None, (101,)),
        (50, (50, 101)),
    ],
)
def test_simulate_bm_matrix_shape_and_initial_condition(n_paths_arg, expected_shape):
    t, W = simulate_bm_matrix(T=1.0, N=100, n_paths=n_paths_arg)
    assert W.shape == expected_shape
    np.testing.assert_array_equal(W[..., 0], 0.0)


@pytest.mark.parametrize(
    "n_paths_arg, expected_shape",
    [
        (None, (101,)),
        (50, (50, 101)),
    ],
)
def test_simulate_drifted_bm_matrix_shape_and_initial_condition(
    n_paths_arg, expected_shape
):
    t, B = simulate_drifted_bm_matrix(
        T=1.0, N=100, x0=x0, mu=mu, sigma=sigma, n_paths=n_paths_arg
    )
    assert B.shape == expected_shape
    np.testing.assert_array_equal(B[..., 0], x0)


@pytest.mark.parametrize("n_paths_arg", [None, 50])
def test_simulate_drifted_bm_deterministic_degen(n_paths_arg):
    t, B = simulate_drifted_bm_matrix(
        T=1.0, N=100, x0=x0, mu=mu, sigma=0.0, n_paths=n_paths_arg
    )
    theory_solution = x0 + mu * t
    expected_matrix = np.broadcast_to(theory_solution, B.shape)

    np.testing.assert_allclose(B, expected_matrix, atol=1e-14)


@pytest.mark.statistical
def test_simulate_drifted_bm_matrix_terminal_mean():
    t, B = simulate_drifted_bm_matrix(
        T=T, N=N, x0=0.0, mu=0.0, sigma=1.0, n_paths=10000, seed=42
    )
    terminal_vector = B[:, -1]
    terminal_emp_mean = np.mean(terminal_vector, axis=0)
    np.testing.assert_allclose(
        actual=terminal_emp_mean, desired=x0 + mu * T, atol=0.0258
    )


@pytest.mark.statistical
def test_simulate_drifted_bm_matrix_terminal_variance():
    t, B = simulate_drifted_bm_matrix(
        T=T, N=N, x0=0.0, mu=0.0, sigma=1.0, n_paths=10000, seed=42
    )
    terminal_vector = B[:, -1]
    terminal_emp_var = np.var(terminal_vector, axis=0)
    sigma2 = sigma**2 * T
    emp_std = sigma2 * np.sqrt(2 / 10000)
    np.testing.assert_allclose(
        actual=terminal_emp_var, desired=sigma**2 * T, atol=2.58 * emp_std
    )
