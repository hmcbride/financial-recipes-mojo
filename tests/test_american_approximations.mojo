from std.math import exp, pow, sqrt
from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.black_scholes import option_price_call_black_scholes, option_price_put_black_scholes
from finrecipes.american_approximations import (
    option_price_american_put_approximated_johnson,
    option_price_american_put_approximated_geske_johnson,
    option_price_american_put_approximated_geske_johnson_corrected,
    option_price_american_call_approximated_baw,
    option_price_american_call_approximated_baw_corrected,
    option_price_american_call_approximated_bjerksund_stensland,
    option_price_american_call_approximated_bjerksund_stensland_corrected,
    option_price_american_put_approximated_bjerksund_stensland,
    option_price_american_put_approximated_bjerksund_stensland_corrected,
)

# Chapter 15 prints two examples: the Johnson put (0.11582) and the
# Barone-Adesi Whaley call (5.74339). Both are asserted below.
#
# The listings of the other approximations have no printed output, and
# C++ Code 15.2 - 15.5 depart from the chapter's formulas (see the docstring
# of finrecipes/american_approximations.mojo). The tests therefore do two
# things:
#  - the routines "as listed" are pinned to the values obtained by compiling
#    the book's C++ listings (with the routines of chapters 9 and 11 and
#    appendix A, and N3 integrated by a 200000 panel Simpson rule);
#  - the `_corrected` routines are compared with values published for the
#    original approximations and with a binomial tree for American options
#    (the method of chapter 12), written out below so that the test does not
#    depend on other modules.


def american_binomial(
    is_call: Bool, S: Float64, X: Float64, r: Float64, y: Float64, sigma: Float64, t: Float64, steps: Int
) -> Float64:
    """Binomial price of an American option on an underlying with continuous payout y."""
    var delta_t = t / Float64(steps)
    var Rinv = exp(-r * delta_t)
    var u = exp(sigma * sqrt(delta_t))
    var uu = u * u
    var d = 1.0 / u
    var p_up = (exp((r - y) * delta_t) - d) / (u - d)
    var p_down = 1.0 - p_up
    var prices = List[Float64](length=steps + 1, fill=0.0)
    var values = List[Float64](length=steps + 1, fill=0.0)
    prices[0] = S * pow(d, Float64(steps))
    for i in range(1, steps + 1):
        prices[i] = uu * prices[i - 1]
    for i in range(steps + 1):
        if is_call:
            values[i] = max(0.0, prices[i] - X)
        else:
            values[i] = max(0.0, X - prices[i])
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            values[i] = (p_up * values[i + 1] + p_down * values[i]) * Rinv
            prices[i] = d * prices[i + 1]
            if is_call:
                values[i] = max(values[i], prices[i] - X)
            else:
                values[i] = max(values[i], X - prices[i])
    return values[0]


def test_johnson_book_example() raises:
    # C++ Code 15.1. Book output: American put price using Johnson approximation = 0.11582
    var P = option_price_american_put_approximated_johnson(1.1, 1.0, 0.125, 0.5, 1.0)
    assert_almost_equal(P, 0.11582, atol=1e-5)
    # the binomial price is 0.1159
    assert_almost_equal(P, american_binomial(False, 1.1, 1.0, 0.125, 0.0, 0.5, 1.0, 1000), atol=1e-3)


def test_johnson_against_binomial() raises:
    # at the money puts with S = X = 40 and r = 0.0488: the approximation is
    # within two cents of the binomial price and never below the European put
    var sigmas: List[Float64] = [0.2, 0.3, 0.4]
    var times: List[Float64] = [0.0833, 0.3333, 0.5833]
    for i in range(3):
        for j in range(3):
            var P = option_price_american_put_approximated_johnson(40.0, 40.0, 0.0488, sigmas[i], times[j])
            var binomial = american_binomial(False, 40.0, 40.0, 0.0488, 0.0, sigmas[i], times[j], 1000)
            assert_almost_equal(P, binomial, atol=0.02)
            assert_true(P >= option_price_put_black_scholes(40.0, 40.0, 0.0488, sigmas[i], times[j]))


def test_geske_johnson_as_listed() raises:
    # C++ Code 15.2 as printed; expected values from compiling the listing
    assert_almost_equal(
        option_price_american_put_approximated_geske_johnson(40.0, 40.0, 0.0488, 0.2, 0.3333), 1.534904, atol=1e-5
    )
    assert_almost_equal(
        option_price_american_put_approximated_geske_johnson(1.1, 1.0, 0.125, 0.5, 1.0), 0.107783, atol=1e-5
    )
    assert_almost_equal(
        option_price_american_put_approximated_geske_johnson(100.0, 100.0, 0.1, 0.25, 1.0), 5.751703, atol=1e-5
    )
    # The listing is not the Geske-Johnson approximation: for this in the
    # money put it returns 6.50 where the American put is worth 5.09.
    var listed = option_price_american_put_approximated_geske_johnson(40.0, 45.0, 0.0488, 0.2, 0.3333)
    assert_almost_equal(listed, 6.498487, atol=1e-5)
    assert_true(listed > american_binomial(False, 40.0, 45.0, 0.0488, 0.0, 0.2, 0.3333, 1000) + 1.0)


def test_geske_johnson_corrected() raises:
    # The parameter grid of the tables in Geske and Johnson (1984): S = 40,
    # r = 0.0488, X = 35, 40, 45, sigma = 0.2, 0.3, 0.4 and T = 1, 4, 7
    # months. The three point extrapolation is within 2.2 cents of a 2000
    # step binomial tree on the whole grid (within 3 cents of the 1000 step
    # tree used here).
    var strikes: List[Float64] = [35.0, 40.0, 45.0]
    var sigmas: List[Float64] = [0.2, 0.3, 0.4]
    var times: List[Float64] = [0.0833, 0.3333, 0.5833]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                var P = option_price_american_put_approximated_geske_johnson_corrected(
                    40.0, strikes[i], 0.0488, sigmas[j], times[k]
                )
                var binomial = american_binomial(False, 40.0, strikes[i], 0.0488, 0.0, sigmas[j], times[k], 1000)
                assert_almost_equal(P, binomial, atol=0.03)
                assert_true(P >= option_price_put_black_scholes(40.0, strikes[i], 0.0488, sigmas[j], times[k]))
    # the parameters of the book's Johnson example (binomial 0.1159) and of
    # the chapter 14 examples (binomial 6.556)
    assert_almost_equal(
        option_price_american_put_approximated_geske_johnson_corrected(1.1, 1.0, 0.125, 0.5, 1.0),
        american_binomial(False, 1.1, 1.0, 0.125, 0.0, 0.5, 1.0, 1000),
        atol=2e-3,
    )
    assert_almost_equal(
        option_price_american_put_approximated_geske_johnson_corrected(100.0, 100.0, 0.1, 0.25, 1.0),
        american_binomial(False, 100.0, 100.0, 0.1, 0.0, 0.25, 1.0, 1000),
        atol=0.03,
    )


def test_baw_book_example() raises:
    # C++ Code 15.3. Book output: Call price using Barone-Adesi Whaley approximation = 5.74339
    assert_almost_equal(
        option_price_american_call_approximated_baw(100.0, 100.0, 0.08, -0.04, 0.20, 0.25), 5.74339, atol=1e-5
    )


def test_baw_corrected() raises:
    # The book's example is taken from Barone-Adesi and Whaley (1987):
    # X = 100, sigma = 0.20, r = 0.08, cost of carry b = -0.04, T = 0.25.
    # Their table gives the quadratic approximation for S = 80, 90, 100, 110,
    # 120 as 0.03, 0.59, 3.52, 10.31, 20.00. (The listing's 5.74339 for
    # S = 100 is the European price for a payout rate of -0.04, 5.63797, plus
    # an exercise premium computed at the seed value of the critical price,
    # because its Newton iteration does not converge.)
    var prices: List[Float64] = [80.0, 90.0, 100.0, 110.0, 120.0]
    var published: List[Float64] = [0.03, 0.59, 3.52, 10.31, 20.00]
    for i in range(5):
        var C = option_price_american_call_approximated_baw_corrected(prices[i], 100.0, 0.08, -0.04, 0.20, 0.25)
        assert_almost_equal(C, published[i], atol=0.005)
        # a cost of carry of -0.04 with r = 0.08 is a payout rate of 0.12
        var binomial = american_binomial(True, prices[i], 100.0, 0.08, 0.12, 0.20, 0.25, 1000)
        assert_almost_equal(C, binomial, atol=0.05)
    # a longer, more volatile option with positive cost of carry
    for i in range(5):
        var C = option_price_american_call_approximated_baw_corrected(prices[i], 100.0, 0.08, 0.04, 0.40, 0.5)
        var binomial = american_binomial(True, prices[i], 100.0, 0.08, 0.04, 0.40, 0.5, 1000)
        assert_almost_equal(C, binomial, atol=0.03)
    # without payout (b = r) the American call is the Black Scholes call
    assert_almost_equal(
        option_price_american_call_approximated_baw_corrected(100.0, 100.0, 0.08, 0.08, 0.20, 0.25),
        option_price_call_black_scholes(100.0, 100.0, 0.08, 0.20, 0.25),
        atol=1e-10,
    )


def test_bjerksund_stensland_as_listed() raises:
    # C++ Code 15.4 and 15.5 as printed; expected values from compiling the listings
    var listed = option_price_american_call_approximated_bjerksund_stensland(42.0, 40.0, 0.04, -0.04, 0.35, 0.75)
    assert_almost_equal(listed, 7.487726, atol=1e-5)
    # the American call is worth 5.31 here; the listing returns its European
    # "lower bound", which is priced for a payout rate of -0.04 instead of 0.08
    assert_true(listed > american_binomial(True, 42.0, 40.0, 0.04, 0.08, 0.35, 0.75, 1000) + 2.0)
    assert_almost_equal(
        option_price_american_call_approximated_bjerksund_stensland(1.05, 1.0, 0.04, -0.04, 0.35, 0.75),
        0.187193,
        atol=1e-6,
    )
    # the put, where the binomial values are 6.556, 7.751 and 6.244
    assert_almost_equal(
        option_price_american_put_approximated_bjerksund_stensland(100.0, 100.0, 0.1, 0.0, 0.25, 1.0),
        16.550817,
        atol=1e-5,
    )
    assert_almost_equal(
        option_price_american_put_approximated_bjerksund_stensland(100.0, 100.0, 0.1, 0.05, 0.25, 1.0),
        9.462496,
        atol=1e-5,
    )
    assert_almost_equal(
        option_price_american_put_approximated_bjerksund_stensland(40.0, 45.0, 0.0488, 0.0, 0.3, 0.5833),
        7.855691,
        atol=1e-5,
    )


def test_bjerksund_stensland_call_corrected() raises:
    # The worked example of the 1993 approximation in Haug, "The Complete
    # Guide to Option Pricing Formulas": S = 42, X = 40, T = 0.75, r = 0.04,
    # b = -0.04, sigma = 0.35 gives 5.2704.
    var C = option_price_american_call_approximated_bjerksund_stensland_corrected(
        42.0, 40.0, 0.04, -0.04, 0.35, 0.75
    )
    assert_almost_equal(C, 5.2704, atol=1e-4)
    # the price scales with the units of S and K
    var C_unit = option_price_american_call_approximated_bjerksund_stensland_corrected(
        1.05, 1.0, 0.04, -0.04, 0.35, 0.75
    )
    assert_almost_equal(40.0 * C_unit, C, atol=1e-9)
    # The approximation values a feasible exercise strategy, so it is a lower
    # bound: here it is within 1.5% of the binomial price and not above it
    # (0.01 allows for the error of the 1000 step tree).
    var prices: List[Float64] = [80.0, 90.0, 100.0, 110.0, 120.0]
    for i in range(5):
        C = option_price_american_call_approximated_bjerksund_stensland_corrected(
            prices[i], 100.0, 0.08, -0.04, 0.20, 0.25
        )
        var binomial = american_binomial(True, prices[i], 100.0, 0.08, 0.12, 0.20, 0.25, 1000)
        assert_true(C <= binomial + 0.01)
        assert_almost_equal(C, binomial, atol=0.015 * binomial + 0.001)
        C = option_price_american_call_approximated_bjerksund_stensland_corrected(
            prices[i], 100.0, 0.08, 0.04, 0.40, 0.5
        )
        binomial = american_binomial(True, prices[i], 100.0, 0.08, 0.04, 0.40, 0.5, 1000)
        assert_true(C <= binomial + 0.01)
        assert_almost_equal(C, binomial, atol=0.015 * binomial + 0.001)
    # far in the money the option is exercised at once
    assert_almost_equal(
        option_price_american_call_approximated_bjerksund_stensland_corrected(200.0, 100.0, 0.08, -0.04, 0.20, 0.25),
        100.0,
        atol=1e-12,
    )
    # without payout (b = r) the American call is the Black Scholes call
    assert_almost_equal(
        option_price_american_call_approximated_bjerksund_stensland_corrected(100.0, 100.0, 0.08, 0.08, 0.20, 0.25),
        option_price_call_black_scholes(100.0, 100.0, 0.08, 0.20, 0.25),
        atol=1e-10,
    )


def test_bjerksund_stensland_put_corrected() raises:
    # puts on an underlying without payout, on the grid used for Geske-Johnson
    var strikes: List[Float64] = [35.0, 40.0, 45.0]
    var sigmas: List[Float64] = [0.2, 0.3, 0.4]
    var times: List[Float64] = [0.0833, 0.3333, 0.5833]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                var P = option_price_american_put_approximated_bjerksund_stensland_corrected(
                    40.0, strikes[i], 0.0488, 0.0, sigmas[j], times[k]
                )
                var binomial = american_binomial(False, 40.0, strikes[i], 0.0488, 0.0, sigmas[j], times[k], 1000)
                assert_true(P <= binomial + 0.01)
                assert_almost_equal(P, binomial, atol=0.025 * binomial + 0.001)
                assert_true(P >= option_price_put_black_scholes(40.0, strikes[i], 0.0488, sigmas[j], times[k]))
    # the parameters of the chapter 14 examples, without and with a payout
    # (binomial 6.556 and 7.751)
    var P = option_price_american_put_approximated_bjerksund_stensland_corrected(100.0, 100.0, 0.1, 0.0, 0.25, 1.0)
    assert_almost_equal(P, american_binomial(False, 100.0, 100.0, 0.1, 0.0, 0.25, 1.0, 1000), atol=0.1)
    P = option_price_american_put_approximated_bjerksund_stensland_corrected(100.0, 100.0, 0.1, 0.05, 0.25, 1.0)
    assert_almost_equal(P, american_binomial(False, 100.0, 100.0, 0.1, 0.05, 0.25, 1.0, 1000), atol=0.1)
    # far in the money the put is exercised at once
    assert_almost_equal(
        option_price_american_put_approximated_bjerksund_stensland_corrected(40.0, 100.0, 0.1, 0.0, 0.25, 1.0),
        60.0,
        atol=1e-12,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
