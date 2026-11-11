from std.testing import TestSuite, assert_almost_equal
from finrecipes.binomial import (
    option_price_call_american_binomial,
    option_price_put_american_binomial,
    option_price_delta_american_call_binomial,
)
from finrecipes.generic_binomial import (
    option_price_generic_binomial,
    option_price_delta_generic_binomial,
    payoff_call,
    payoff_put,
    payoff_binary_call,
    payoff_binary_put,
)


def test_payoffs() raises:
    assert_almost_equal(payoff_call(110.0, 100.0), 10.0, atol=1e-12)
    assert_almost_equal(payoff_call(90.0, 100.0), 0.0, atol=1e-12)
    assert_almost_equal(payoff_put(110.0, 100.0), 0.0, atol=1e-12)
    assert_almost_equal(payoff_put(90.0, 100.0), 10.0, atol=1e-12)
    assert_almost_equal(payoff_binary_call(100.0, 100.0), 1.0, atol=1e-12)
    assert_almost_equal(payoff_binary_call(99.0, 100.0), 0.0, atol=1e-12)
    assert_almost_equal(payoff_binary_put(100.0, 100.0), 1.0, atol=1e-12)
    assert_almost_equal(payoff_binary_put(101.0, 100.0), 0.0, atol=1e-12)


def test_american_call_and_put() raises:
    # Book example, section 17.1 (the program uses sigma = 0.25)
    assert_almost_equal(
        option_price_generic_binomial(100.0, 100.0, payoff_call, 0.1, 0.25, 1.0, 100), 14.9505, atol=1e-4
    )
    assert_almost_equal(
        option_price_generic_binomial(100.0, 100.0, payoff_put, 0.1, 0.25, 1.0, 100), 6.54691, atol=1e-5
    )


def test_matches_chapter_12_routines() raises:
    # C++ Code 17.1 is the American call of chapter 12
    assert_almost_equal(
        option_price_generic_binomial(100.0, 100.0, payoff_call, 0.1, 0.25, 1.0, 100),
        option_price_call_american_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100),
        atol=1e-12,
    )
    assert_almost_equal(
        option_price_generic_binomial(100.0, 100.0, payoff_put, 0.1, 0.25, 1.0, 100),
        option_price_put_american_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100),
        atol=1e-12,
    )


def test_binary_option() raises:
    # Book example, section 17.1: K = 120
    assert_almost_equal(
        option_price_generic_binomial(100.0, 120.0, payoff_binary_call, 0.1, 0.25, 1.0, 100), 0.498858, atol=1e-6
    )


def test_delta() raises:
    # the generic delta with a call payoff is the delta of C++ Code 12.3,
    # whose value is printed in the chapter 12 example
    var delta = option_price_delta_generic_binomial(100.0, 100.0, payoff_call, 0.1, 0.25, 1.0, 100)
    assert_almost_equal(delta, 0.699792, atol=1e-6)
    assert_almost_equal(
        delta, option_price_delta_american_call_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100), atol=1e-12
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
