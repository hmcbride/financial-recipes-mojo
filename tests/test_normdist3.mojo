from std.math import asin, sqrt
from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.normdist import N, N2
from finrecipes.normdist3 import N3, N3Parms, f3

# The book prints no example for C++ Code C.1, so N3 is checked against
# closed forms. N3 integrates the four point bivariate approximation N2 of
# Appendix A, which is good to about four decimals, hence atol=1e-4 where
# N2 enters with a nonzero correlation.

comptime PI: Float64 = 3.141592653589793238462643


def test_independent_variables() raises:
    # with zero correlations the probability is the product of the marginals
    assert_almost_equal(N3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.125, atol=1e-6)
    assert_almost_equal(N3(0.3, -0.2, 0.5, 0.0, 0.0, 0.0), N(0.3) * N(-0.2) * N(0.5), atol=1e-6)
    assert_almost_equal(N3(-1.0, 1.5, 0.7, 0.0, 0.0, 0.0), N(-1.0) * N(1.5) * N(0.7), atol=1e-6)


def test_orthant_probability() raises:
    # P(x1<0, x2<0, x3<0) = 1/8 + (asin rho12 + asin rho13 + asin rho23) / (4 pi)
    assert_almost_equal(N3(0.0, 0.0, 0.0, 0.5, 0.5, 0.5), 0.25, atol=1e-4)
    var rho12: Float64 = 0.2
    var rho13: Float64 = 0.4
    var rho23: Float64 = -0.1
    var expected = 0.125 + (asin(rho12) + asin(rho13) + asin(rho23)) / (4.0 * PI)
    assert_almost_equal(N3(0.0, 0.0, 0.0, rho12, rho13, rho23), expected, atol=1e-4)
    # the correlations used by the Geske-Johnson put: rho12, -rho13, -rho23
    rho12 = 1.0 / sqrt(Float64(2.0))
    rho13 = -1.0 / sqrt(Float64(3.0))
    rho23 = -sqrt(Float64(2.0) / 3.0)
    expected = 0.125 + (asin(rho12) + asin(rho13) + asin(rho23)) / (4.0 * PI)
    assert_almost_equal(N3(0.0, 0.0, 0.0, rho12, rho13, rho23), expected, atol=1e-4)


def test_marginal_limits() raises:
    # a limit far out in one variable leaves the bivariate normal of the others
    assert_almost_equal(N3(0.5, 0.5, 8.0, 0.3, 0.1, 0.2), N2(0.5, 0.5, 0.3), atol=1e-4)
    assert_almost_equal(N3(8.0, 0.5, -0.4, 0.3, 0.1, 0.2), N2(0.5, -0.4, 0.2), atol=1e-4)
    assert_almost_equal(N3(0.5, 8.0, -0.4, 0.3, 0.1, 0.2), N2(0.5, -0.4, 0.1), atol=1e-4)
    # and far out in two variables leaves the univariate normal
    assert_almost_equal(N3(8.0, 8.0, 0.25, 0.3, 0.1, 0.2), N(0.25), atol=1e-6)
    # a limit far down in any variable gives probability zero
    assert_almost_equal(N3(0.5, 0.5, -8.0, 0.3, 0.1, 0.2), 0.0, atol=1e-6)
    assert_almost_equal(N3(-8.0, 0.5, 0.5, 0.3, 0.1, 0.2), 0.0, atol=1e-6)


def test_symmetry_and_monotonicity() raises:
    # exchanging variables 1 and 2 exchanges h, k and rho13, rho23
    assert_almost_equal(N3(1.0, 0.5, -0.3, 0.2, 0.4, -0.1), N3(0.5, 1.0, -0.3, 0.2, -0.1, 0.4), atol=1e-4)
    # the probability increases in each limit and stays inside [0, 1]
    var p = N3(1.0, 0.5, -0.3, 0.2, 0.4, -0.1)
    assert_true(p > 0.0 and p < 1.0)
    assert_true(N3(1.2, 0.5, -0.3, 0.2, 0.4, -0.1) > p)
    assert_true(N3(1.0, 0.7, -0.3, 0.2, 0.4, -0.1) > p)
    assert_true(N3(1.0, 0.5, -0.1, 0.2, 0.4, -0.1) > p)
    assert_true(p < N(-0.3))


def test_reference_values() raises:
    # the same integral evaluated with a 200000 panel Simpson rule in C++
    # (the listing with the GSL call replaced); agreement shows the adaptive
    # quadrature meets its 1e-7 tolerance
    assert_almost_equal(N3(1.0, 0.5, -0.3, 0.2, 0.4, -0.1), 0.2383645729, atol=1e-6)
    var rho12 = 1.0 / sqrt(Float64(2.0))
    var rho13 = -1.0 / sqrt(Float64(3.0))
    var rho23 = -sqrt(Float64(2.0) / 3.0)
    assert_almost_equal(N3(0.3, -0.2, 0.5, rho12, rho13, rho23), 0.1327891748, atol=1e-6)


def test_integrand() raises:
    # f3 is the density of z times the bivariate normal of the other two
    var parms = N3Parms(0.0, 0.0, 0.0, 0.0, 0.0)
    assert_almost_equal(f3(0.0, parms), 0.3989423 * 0.25, atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
