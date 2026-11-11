from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.black_scholes import option_price_put_black_scholes
from finrecipes.binomial import option_price_put_american_binomial
from finrecipes.bond_options import (
    bond_option_price_put_zero_black_scholes,
    bond_option_price_put_coupon_bond_black_scholes,
    bond_option_price_put_american_binomial,
)


def test_put_on_zero_coupon_bond() raises:
    # Book example, section 20.2: B=100, K=100, r=0.05, sigma=0.1, time=1
    assert_almost_equal(bond_option_price_put_zero_black_scholes(100.0, 100.0, 0.05, 0.1, 1.0), 1.92791, atol=1e-5)
    # same formula as the stock option put of chapter 9
    assert_almost_equal(
        bond_option_price_put_zero_black_scholes(100.0, 100.0, 0.05, 0.1, 1.0),
        option_price_put_black_scholes(100.0, 100.0, 0.05, 0.1, 1.0),
        atol=1e-12,
    )


def test_put_on_coupon_bond() raises:
    # the program pays a coupon of 1 at time 0.5
    var coupon_times: List[Float64] = [0.5]
    var coupons: List[Float64] = [1.0]
    assert_almost_equal(
        bond_option_price_put_coupon_bond_black_scholes(100.0, 100.0, 0.05, 0.1, 1.0, coupon_times, coupons),
        2.22852,
        atol=1e-5,
    )


def test_coupons_after_option_maturity_are_ignored() raises:
    var coupon_times: List[Float64] = [1.5, 2.0]
    var coupons: List[Float64] = [1.0, 1.0]
    assert_almost_equal(
        bond_option_price_put_coupon_bond_black_scholes(100.0, 100.0, 0.05, 0.1, 1.0, coupon_times, coupons),
        bond_option_price_put_zero_black_scholes(100.0, 100.0, 0.05, 0.1, 1.0),
        atol=1e-12,
    )


def test_american_put_binomial() raises:
    var american = bond_option_price_put_american_binomial(100.0, 100.0, 0.05, 0.1, 1.0, 100)
    assert_almost_equal(american, 2.43282, atol=1e-5)
    # the American put is worth more than the European one
    assert_true(american > bond_option_price_put_zero_black_scholes(100.0, 100.0, 0.05, 0.1, 1.0))
    # and it is the stock option routine of chapter 12 with the bond as underlying
    assert_almost_equal(american, option_price_put_american_binomial(100.0, 100.0, 0.05, 0.1, 1.0, 100), atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
