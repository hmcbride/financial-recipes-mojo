from std.math import exp
from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.credit_risk import debt_value_merton


def test_debt_value() raises:
    # Book example, section 21.1: V=100, F=90, r=0.05, T=1, sigma=0.25
    assert_almost_equal(debt_value_merton(100.0, 90.0, 0.05, 0.25, 1.0), 81.8592, atol=1e-4)


def test_debt_is_worth_less_than_risk_free_debt() raises:
    var risk_free = exp(-0.05 * 1.0) * 90.0
    var risky = debt_value_merton(100.0, 90.0, 0.05, 0.25, 1.0)
    assert_true(risky < risk_free)
    # more volatile firm value: larger put value, lower debt value
    assert_true(debt_value_merton(100.0, 90.0, 0.05, 0.5, 1.0) < risky)
    # firm value far above the face value: debt is close to risk free
    assert_almost_equal(debt_value_merton(1000.0, 90.0, 0.05, 0.25, 1.0), risk_free, atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
