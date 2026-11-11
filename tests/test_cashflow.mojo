from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_true, assert_false
from finrecipes.cashflow import (
    cash_flow_pv_discrete,
    cash_flow_irr_discrete,
    cash_flow_unique_irr,
    cash_flow_pv,
    sgn,
)


def test_pv_discrete() raises:
    # Book example, C++ Code 3.1: investment of 100, 75 in each of the next two years
    var cflows: List[Float64] = [-100.0, 75.0, 75.0]
    var times: List[Float64] = [0.0, 1.0, 2.0]
    assert_almost_equal(cash_flow_pv_discrete(times, cflows, 0.1), 30.1653, atol=1e-4)


def test_pv_and_irr_discrete() raises:
    # Book example, C++ Code 3.2
    var cflows: List[Float64] = [-100.0, 10.0, 110.0]
    var times: List[Float64] = [0.0, 1.0, 2.0]
    assert_almost_equal(cash_flow_pv_discrete(times, cflows, 0.05), 9.29705, atol=1e-5)
    assert_almost_equal(cash_flow_irr_discrete(times, cflows), 0.1, atol=1e-6)


def test_irr_makes_pv_zero() raises:
    # Root needs several bisection steps; the routine stops within 1e-5 of it.
    var cflows: List[Float64] = [-100.0, 30.0, 40.0, 50.0]
    var times: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var irr = cash_flow_irr_discrete(times, cflows)
    assert_almost_equal(irr, 0.0890, atol=1e-4)
    assert_almost_equal(cash_flow_pv_discrete(times, cflows, irr), 0.0, atol=2e-3)


def test_irr_bracket_is_expanded() raises:
    # IRR of 50 percent lies outside the initial bracket [0, 0.2]
    var cflows: List[Float64] = [-100.0, 150.0]
    var times: List[Float64] = [0.0, 1.0]
    assert_almost_equal(cash_flow_irr_discrete(times, cflows), 0.5, atol=1e-5)


def test_irr_errors() raises:
    var times: List[Float64] = [0.0, 1.0, 2.0]
    # different number of times and amounts
    assert_almost_equal(cash_flow_irr_discrete(times, [-100.0, 10.0]), -1e30, rtol=1e-12)
    # only positive cash flows: no root to bracket
    assert_almost_equal(cash_flow_irr_discrete(times, [100.0, 10.0, 110.0]), -1e30, rtol=1e-12)


def test_sgn() raises:
    assert_equal(sgn(2.5), 1)
    assert_equal(sgn(0.0), 1)
    assert_equal(sgn(-0.1), -1)


def test_unique_irr() raises:
    var times: List[Float64] = [0.0, 1.0, 2.0]
    # one sign change (the cash flows of the left hand figure in section 3.2.1)
    assert_true(cash_flow_unique_irr(times, [-100.0, 10.0, 100.0]))
    assert_true(cash_flow_unique_irr(times, [-100.0, 10.0, 110.0]))
    # no sign change: no irr
    assert_false(cash_flow_unique_irr(times, [100.0, 10.0, 110.0]))
    # two sign changes, but the aggregated cash flows -100, 101, 1 change sign
    # once, so the Norstrom check passes (right hand figure in section 3.2.1)
    assert_true(cash_flow_unique_irr(times, [-100.0, 201.0, -100.0]))
    # two sign changes, aggregated cash flows -100, 200, -50 change sign twice
    assert_false(cash_flow_unique_irr(times, [-100.0, 300.0, -250.0]))


def test_pv_continuous() raises:
    # C++ Code 3.4 has no book example; expected values are
    # -100 + 75 exp(-0.1) + 75 exp(-0.2) and -100 + 10 exp(-0.05) + 110 exp(-0.1)
    var times: List[Float64] = [0.0, 1.0, 2.0]
    assert_almost_equal(cash_flow_pv(times, [-100.0, 75.0, 75.0], 0.1), 29.2676128335, atol=1e-9)
    assert_almost_equal(cash_flow_pv(times, [-100.0, 10.0, 110.0], 0.05), 9.0444102290, atol=1e-9)
    # zero interest: the sum of the cash flows
    assert_almost_equal(cash_flow_pv(times, [-100.0, 75.0, 75.0], 0.0), 50.0, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
