from std.testing import TestSuite, assert_almost_equal
from finrecipes.futures import futures_price


def test_futures_price() raises:
    # C++ Code 7.1 example: futures price = 105.127
    assert_almost_equal(futures_price(100.0, 0.10, 0.5), 105.127, atol=1e-3)


def test_futures_price_limits() raises:
    # no time to maturity or no interest: the futures price is the spot price
    assert_almost_equal(futures_price(100.0, 0.10, 0.0), 100.0, atol=1e-12)
    assert_almost_equal(futures_price(100.0, 0.0, 0.5), 100.0, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
