from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.binomial import option_price_put_american_binomial
from finrecipes.trinomial import option_price_put_american_trinomial


def test_american_put() raises:
    # Book example, section 18.2: S=100, K=100, r=0.1, q=0, sigma=0.25, 1 year, 100 steps
    assert_almost_equal(option_price_put_american_trinomial(100.0, 100.0, 0.1, 0.0, 0.25, 1.0, 100), 6.52719, atol=1e-5)


def test_close_to_binomial() raises:
    # the trinomial and binomial approximations price the same option
    var trinomial = option_price_put_american_trinomial(100.0, 100.0, 0.1, 0.0, 0.25, 1.0, 100)
    var binomial = option_price_put_american_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100)
    assert_almost_equal(trinomial, binomial, atol=0.05)


def test_payout_raises_put_price() raises:
    var no_payout = option_price_put_american_trinomial(100.0, 100.0, 0.1, 0.0, 0.25, 1.0, 100)
    var payout = option_price_put_american_trinomial(100.0, 100.0, 0.1, 0.05, 0.25, 1.0, 100)
    assert_true(payout > no_payout)


def test_single_step() raises:
    # one step by hand: nodes S*d, S, S*u with probabilities p_d, p_m, p_u
    var price = option_price_put_american_trinomial(100.0, 110.0, 0.1, 0.0, 0.25, 1.0, 1)
    # deep enough in the money that immediate exercise is optimal
    assert_almost_equal(price, 10.0, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
