from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.black_scholes import option_price_call_black_scholes, option_price_put_black_scholes
from finrecipes.matrix import mat_solve, mat_zeros
from finrecipes.finite_differences import (
    mat_solve_tridiagonal,
    option_price_put_european_finite_diff_explicit,
    option_price_put_american_finite_diff_explicit,
    option_price_call_european_finite_diff_explicit,
    option_price_call_american_finite_diff_explicit,
    option_price_put_american_finite_diff_implicit,
    option_price_put_european_finite_diff_implicit,
)


def test_explicit_puts() raises:
    # Book example, section 13.3: 20 steps in S, 11 steps in time
    assert_almost_equal(
        option_price_put_european_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 20, 11), 4.03667, atol=1e-5
    )
    assert_almost_equal(
        option_price_put_american_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 20, 11), 4.25085, atol=1e-5
    )


def test_odd_number_of_price_steps_is_rounded_up() raises:
    assert_almost_equal(
        option_price_put_european_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 19, 11),
        option_price_put_european_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 20, 11),
        atol=1e-12,
    )
    assert_almost_equal(
        option_price_put_american_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 19, 11),
        option_price_put_american_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 20, 11),
        atol=1e-12,
    )


def test_explicit_converges_to_black_scholes() raises:
    # on a fine (and stable) grid the European prices approach Black Scholes
    var bs_put = option_price_put_black_scholes(50.0, 50.0, 0.1, 0.4, 0.4167)
    assert_almost_equal(
        option_price_put_european_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 100, 1000), bs_put, atol=1e-2
    )
    var bs_call = option_price_call_black_scholes(50.0, 50.0, 0.1, 0.4, 0.4167)
    assert_almost_equal(
        option_price_call_european_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 100, 1000), bs_call, atol=1e-2
    )


def test_explicit_calls() raises:
    # an American call on a stock without dividends is never exercised early
    var european = option_price_call_european_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 20, 11)
    var american = option_price_call_american_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 20, 11)
    assert_almost_equal(american, european, atol=1e-9)
    assert_almost_equal(european, option_price_call_black_scholes(50.0, 50.0, 0.1, 0.4, 0.4167), atol=0.1)


def test_implicit_puts() raises:
    # Book example, section 13.8: 200 steps in S and in time
    assert_almost_equal(option_price_put_black_scholes(50.0, 50.0, 0.1, 0.4, 0.5), 4.35166, atol=1e-5)
    assert_almost_equal(
        option_price_put_european_finite_diff_implicit(50.0, 50.0, 0.1, 0.4, 0.5, 200, 200), 4.34731, atol=1e-5
    )
    assert_almost_equal(
        option_price_put_american_finite_diff_implicit(50.0, 50.0, 0.1, 0.4, 0.5, 200, 200), 4.60064, atol=1e-5
    )


def test_implicit_matlab_example() raises:
    # Matlab output in the same example: P = 4.6006
    assert_almost_equal(
        option_price_put_american_finite_diff_implicit(50.0, 50.0, 0.1, 0.4, 0.5, 200, 200), 4.6006, atol=1e-4
    )


def test_american_put_worth_more_than_european() raises:
    assert_true(
        option_price_put_american_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 20, 11)
        > option_price_put_european_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 20, 11)
    )
    assert_true(
        option_price_put_american_finite_diff_implicit(50.0, 50.0, 0.1, 0.4, 0.5, 20, 20)
        > option_price_put_european_finite_diff_implicit(50.0, 50.0, 0.1, 0.4, 0.5, 20, 20)
    )


def test_tridiagonal_solve_is_mat_solve() raises:
    # the system of the implicit scheme: 50 price steps, sigma=0.4, r=0.1, delta_t=0.01
    var n = 51
    var A = mat_zeros(n, n)
    var B = List[Float64](length=n, fill=0.0)
    A[0][0] = 1.0
    for j in range(1, n - 1):
        var fj = Float64(j)
        A[j][j - 1] = 0.5 * fj * 0.01 * (0.1 - 0.16 * fj)
        A[j][j] = 1.0 + 0.01 * (0.1 + 0.16 * fj * fj)
        A[j][j + 1] = 0.5 * fj * 0.01 * (-0.1 - 0.16 * fj)
    A[n - 1][n - 1] = 1.0
    for m in range(n):
        B[m] = max(0.0, 50.0 - 2.0 * Float64(m))
    var dense = mat_solve(A, B)
    var banded = mat_solve_tridiagonal(A, B)
    for m in range(n):
        assert_true(banded[m] == dense[m])


def test_tridiagonal_solve_with_row_exchanges() raises:
    # the entries below the diagonal are the largest, so every column is pivoted
    var A: List[List[Float64]] = [
        [1.0, 2.0, 0.0, 0.0, 0.0],
        [3.0, 1.0, 2.0, 0.0, 0.0],
        [0.0, 4.0, 1.0, 2.0, 0.0],
        [0.0, 0.0, 5.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 6.0, 1.0],
    ]
    var B: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var dense = mat_solve(A, B)
    var banded = mat_solve_tridiagonal(A, B)
    for m in range(5):
        assert_true(banded[m] == dense[m])
    # and the solution satisfies the equations
    for i in range(5):
        var s: Float64 = 0.0
        for j in range(5):
            s += A[i][j] * banded[j]
        assert_almost_equal(s, B[i], atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
