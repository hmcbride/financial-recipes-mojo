from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.rendleman_bartter import bond_option_price_call_zero_american_rendleman_bartter


def test_call_on_zero_coupon_bond() raises:
    # Book example, section 23.1: K=950, S=0.15, M=0.05, interest=10%,
    # option matures in 4 years, bond in 5 years paying 1000, 100 steps
    assert_almost_equal(
        bond_option_price_call_zero_american_rendleman_bartter(950.0, 4.0, 0.15, 0.05, 0.10, 5.0, 1000.0, 100),
        0.00713661,
        atol=1e-8,
    )


def test_lower_exercise_price_is_worth_more() raises:
    var high = bond_option_price_call_zero_american_rendleman_bartter(950.0, 4.0, 0.15, 0.05, 0.10, 5.0, 1000.0, 100)
    var low = bond_option_price_call_zero_american_rendleman_bartter(900.0, 4.0, 0.15, 0.05, 0.10, 5.0, 1000.0, 100)
    assert_true(low > high)
    assert_true(high > 0.0)


def test_zero_exercise_price_gives_bond_price() raises:
    # with X = 0 the call is the bond itself; without rate uncertainty and
    # drift the bond price is the payment discounted at the short rate
    var price = bond_option_price_call_zero_american_rendleman_bartter(0.0, 4.0, 1e-9, 0.0, 0.10, 5.0, 1000.0, 100)
    assert_almost_equal(price, 606.5307, atol=1e-3)  # 1000 exp(-0.1*5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
