from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_true
from finrecipes.mean_variance import (
    mv_calculate_mean,
    mv_calculate_variance,
    mv_calculate_st_dev,
    mv_calculate_portfolio_given_mean_unconstrained,
    min_variance_frontier,
    min_variance_portfolio,
    mean,
    cov,
    variance,
    stdev,
    sharpe,
    treynor,
    jensen,
)


def test_matlab_equally_weighted_portfolio() raises:
    # Section 6.1 Matlab example: er = 0.096667, sigma = 0.2236
    var e: List[Float64] = [0.1, 0.11, 0.08]
    var V: List[List[Float64]] = [[0.2, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.15]]
    var third: Float64 = 1.0 / 3.0
    var w: List[Float64] = [third, third, third]
    assert_almost_equal(mv_calculate_mean(e, w), 0.096667, atol=1e-6)
    assert_almost_equal(mv_calculate_st_dev(V, w), 0.2236, atol=1e-4)


def test_matlab_minimum_variance_portfolio() raises:
    # Section 6.3 Matlab example (the script that becomes Matlab Code 6.1), r = 0.09
    var e: List[Float64] = [0.1, 0.11, 0.08]
    var V: List[List[Float64]] = [[0.2, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.15]]
    var f = min_variance_frontier(e, V)
    assert_almost_equal(f.a, 2.1333, atol=1e-4)
    assert_almost_equal(f.b, 0.2137, atol=1e-4)
    assert_almost_equal(f.c, 21.667, atol=1e-3)
    assert_almost_equal(f.d, 0.078333, atol=1e-6)
    var g: List[Float64] = [0.021277, -2.680851, 3.659574]
    var h: List[Float64] = [2.1277, 31.9149, -34.0426]
    var w_expected: List[Float64] = [0.2128, 0.1915, 0.5957]
    assert_equal(len(f.g), 3)
    assert_equal(len(f.h), 3)
    var w = f.weights(0.09)
    var w2 = min_variance_portfolio(e, V, 0.09)
    for i in range(3):
        assert_almost_equal(f.g[i], g[i], atol=1e-6)
        assert_almost_equal(f.h[i], h[i], atol=1e-4)
        assert_almost_equal(w[i], w_expected[i], atol=1e-4)
        assert_almost_equal(w2[i], w_expected[i], atol=1e-4)


def test_frontier_properties() raises:
    # g sums to one and has mean 0, h sums to zero and has mean 1 (section 6.3)
    var e: List[Float64] = [0.1, 0.11, 0.08]
    var V: List[List[Float64]] = [[0.2, 0.01, 0.0], [0.01, 0.1, -0.02], [0.0, -0.02, 0.15]]
    var ones: List[Float64] = [1.0, 1.0, 1.0]
    var f = min_variance_frontier(e, V)
    assert_almost_equal(mv_calculate_mean(ones, f.g), 1.0, atol=1e-10)
    assert_almost_equal(mv_calculate_mean(ones, f.h), 0.0, atol=1e-10)
    assert_almost_equal(mv_calculate_mean(e, f.g), 0.0, atol=1e-10)
    assert_almost_equal(mv_calculate_mean(e, f.h), 1.0, atol=1e-10)
    # the Matlab and the C++ routines give the same frontier portfolio
    var w_matlab = min_variance_portfolio(e, V, 0.095)
    var w_cpp = mv_calculate_portfolio_given_mean_unconstrained(e, V, 0.095)
    for i in range(3):
        assert_almost_equal(w_matlab[i], w_cpp[i], atol=1e-10)
    assert_almost_equal(mv_calculate_mean(e, w_cpp), 0.095, atol=1e-10)
    # the global minimum variance portfolio has mean a/c and variance 1/c (section 6.4)
    var w_mvp = f.weights(f.global_minimum_variance_mean())
    assert_almost_equal(mv_calculate_mean(e, w_mvp), f.a / f.c, atol=1e-10)
    assert_almost_equal(mv_calculate_variance(V, w_mvp), f.global_minimum_variance_variance(), atol=1e-10)
    # and no frontier portfolio has a lower variance
    assert_true(mv_calculate_variance(V, w_mvp) < mv_calculate_variance(V, w_cpp))


def test_cpp_mean_variance_calculations() raises:
    # C++ Code 6.1 / 6.2 example: mean 0.075, variance 0.5, stdev 0.707107
    var e: List[Float64] = [0.05, 0.1]
    var V: List[List[Float64]] = [[1.0, 0.0], [0.0, 1.0]]
    var w: List[Float64] = [0.5, 0.5]
    assert_almost_equal(mv_calculate_mean(e, w), 0.075, atol=1e-6)
    assert_almost_equal(mv_calculate_variance(V, w), 0.5, atol=1e-6)
    assert_almost_equal(mv_calculate_st_dev(V, w), 0.707107, atol=1e-6)


def test_cpp_portfolio_given_mean() raises:
    # C++ Code 6.3 / 6.4 example: w1 = 0.5, w2 = 0.5
    var e: List[Float64] = [0.05, 0.1]
    var V: List[List[Float64]] = [[1.0, 0.0], [0.0, 1.0]]
    var w = mv_calculate_portfolio_given_mean_unconstrained(e, V, 0.075)
    assert_equal(len(w), 2)
    assert_almost_equal(w[0], 0.5, atol=1e-6)
    assert_almost_equal(w[1], 0.5, atol=1e-6)


def test_sample_statistics() raises:
    var x: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var y: List[Float64] = [2.0, 4.0, 6.0, 9.0]
    assert_almost_equal(mean(x), 2.5, atol=1e-12)
    assert_almost_equal(variance(x), 5.0 / 3.0, atol=1e-12)
    assert_almost_equal(stdev(x), 1.2909944487358056, atol=1e-12)
    assert_almost_equal(cov(x, y), 11.5 / 3.0, atol=1e-12)


def test_performance_measures() raises:
    # Matlab Code 6.2 - 6.4. The book has no example; the expected values
    # were worked out independently from the definitions.
    var r: List[Float64] = [0.10, 0.05, 0.12, -0.02, 0.08]
    var rm: List[Float64] = [0.08, 0.04, 0.10, -0.01, 0.06]
    var rf: List[Float64] = [0.02, 0.02, 0.03, 0.03, 0.02]
    assert_almost_equal(sharpe(r, rf), 0.7459670749916573, atol=1e-10)
    assert_almost_equal(sharpe(r, 0.02), 0.8426548012644084, atol=1e-10)
    assert_almost_equal(treynor(r, rm, rf), 0.032575163398692805, atol=1e-10)
    assert_almost_equal(treynor(r, rm, 0.02), 0.03567755991285403, atol=1e-10)
    assert_almost_equal(jensen(r, rm, 0.02), 0.0021629213483146115, atol=1e-10)
    # the market itself has beta 1: Treynor is its excess return, Jensen's alpha is 0
    assert_almost_equal(treynor(rm, rm, 0.02), mean(rm) - 0.02, atol=1e-12)
    assert_almost_equal(jensen(rm, rm, 0.02), 0.0, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
