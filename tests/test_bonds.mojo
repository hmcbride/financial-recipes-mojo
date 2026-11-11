from std.testing import TestSuite, assert_almost_equal
from finrecipes.bonds import (
    bonds_price_discrete,
    bonds_yield_to_maturity_discrete,
    bonds_duration_discrete,
    bonds_duration_macaulay_discrete,
    bonds_duration_modified_discrete,
    bonds_convexity_discrete,
    bonds_price,
    bonds_duration,
    bonds_yield_to_maturity,
    bonds_duration_macaulay,
    bonds_convexity,
)

# All the chapter's examples use the same bond: 3 years, face value 100,
# annual coupon of 10 percent, interest rate 9 percent.


def test_price_discrete() raises:
    # Book example, C++ Code 4.1
    var cflows: List[Float64] = [10.0, 10.0, 110.0]
    var times: List[Float64] = [1.0, 2.0, 3.0]
    assert_almost_equal(bonds_price_discrete(times, cflows, 0.09), 102.531, atol=1e-3)


def test_yield_to_maturity_discrete() raises:
    # Book example, C++ Code 4.2
    var cflows: List[Float64] = [10.0, 10.0, 110.0]
    var times: List[Float64] = [1.0, 2.0, 3.0]
    var B = bonds_price_discrete(times, cflows, 0.09)
    assert_almost_equal(B, 102.531, atol=1e-3)
    assert_almost_equal(bonds_yield_to_maturity_discrete(times, cflows, B), 0.09, atol=1e-6)


def test_duration_discrete() raises:
    # Book example, C++ Code 4.3 and 4.4
    var cflows: List[Float64] = [10.0, 10.0, 110.0]
    var times: List[Float64] = [1.0, 2.0, 3.0]
    var B = bonds_price_discrete(times, cflows, 0.09)
    assert_almost_equal(B, 102.531, atol=1e-3)
    assert_almost_equal(bonds_duration_discrete(times, cflows, 0.09), 2.73895, atol=1e-5)
    assert_almost_equal(bonds_duration_macaulay_discrete(times, cflows, B), 2.73895, atol=1e-5)


def test_modified_duration_and_convexity_discrete() raises:
    # Book example, C++ Code 4.5 and 4.6
    var cflows: List[Float64] = [10.0, 10.0, 110.0]
    var times: List[Float64] = [1.0, 2.0, 3.0]
    var B = bonds_price_discrete(times, cflows, 0.09)
    assert_almost_equal(B, 102.531, atol=1e-3)
    assert_almost_equal(bonds_duration_discrete(times, cflows, 0.09), 2.73895, atol=1e-5)
    assert_almost_equal(bonds_duration_modified_discrete(times, cflows, B), 2.5128, atol=1e-4)
    assert_almost_equal(bonds_convexity_discrete(times, cflows, 0.09), 8.93248, atol=1e-5)
    assert_almost_equal(bonds_price_discrete(times, cflows, 0.1), 100.0, atol=1e-4)


def test_price_change_approximation_discrete() raises:
    # The worked calculation after the example: estimate the price at 10 percent
    # from the price at 9 percent. Duration only gives 99.957, duration and
    # convexity 100.0036 (the book rounds D* to 2.51 and Cx to 8.93).
    var cflows: List[Float64] = [10.0, 10.0, 110.0]
    var times: List[Float64] = [1.0, 2.0, 3.0]
    var B = bonds_price_discrete(times, cflows, 0.09)
    var Dmod = bonds_duration_modified_discrete(times, cflows, B)
    var Cx = bonds_convexity_discrete(times, cflows, 0.09)
    var dy: Float64 = 0.01
    assert_almost_equal(B * (1.0 - Dmod * dy), 99.957, atol=5e-3)
    assert_almost_equal(B * (1.0 - Dmod * dy + 0.5 * Cx * dy * dy), 100.0036, atol=5e-3)


def test_continuous() raises:
    # Book example, C++ Code 4.7, 4.8 and 4.10
    var cflows: List[Float64] = [10.0, 10.0, 110.0]
    var times: List[Float64] = [1.0, 2.0, 3.0]
    var B = bonds_price(times, cflows, 0.09)
    assert_almost_equal(B, 101.464, atol=1e-3)
    assert_almost_equal(bonds_duration(times, cflows, 0.09), 2.73753, atol=1e-5)
    assert_almost_equal(bonds_convexity(times, cflows, 0.09), 7.86779, atol=1e-5)
    assert_almost_equal(bonds_price(times, cflows, 0.08), 104.282, atol=1e-3)


def test_yield_and_macaulay_continuous() raises:
    # C++ Code 4.9 has no printed output. With a flat term structure the yield
    # to maturity of a correctly priced bond is the interest rate, so Macaulay
    # duration equals duration (Exercise 4.3).
    var cflows: List[Float64] = [10.0, 10.0, 110.0]
    var times: List[Float64] = [1.0, 2.0, 3.0]
    var B = bonds_price(times, cflows, 0.09)
    assert_almost_equal(bonds_yield_to_maturity(times, cflows, B), 0.09, atol=1e-6)
    assert_almost_equal(bonds_duration_macaulay(times, cflows, B), 2.73753, atol=1e-5)
    # a bond priced at 104.282 yields 8 percent (last line of the book example)
    assert_almost_equal(bonds_yield_to_maturity(times, cflows, 104.282), 0.08, atol=1e-5)


def test_zero_coupon_bond() raises:
    # A zero coupon bond has duration equal to its maturity
    var cflows: List[Float64] = [100.0]
    var times: List[Float64] = [5.0]
    assert_almost_equal(bonds_duration_discrete(times, cflows, 0.07), 5.0, atol=1e-12)
    assert_almost_equal(bonds_duration(times, cflows, 0.07), 5.0, atol=1e-12)
    assert_almost_equal(bonds_convexity(times, cflows, 0.07), 25.0, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
