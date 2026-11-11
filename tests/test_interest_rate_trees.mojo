from std.testing import TestSuite, assert_almost_equal, assert_equal
from finrecipes.interest_rate_trees import (
    interest_rate_trees_gbm_build,
    interest_rate_trees_gbm_value_of_cashflows,
    interest_rate_trees_gbm_value_of_callable_bond,
)


def test_build_tree() raises:
    # Book example, section 24.1: r0=10%, u=1.02, d=0.99
    var tree = interest_rate_trees_gbm_build(0.1, 1.02, 0.99, 3)
    assert_equal(len(tree), 4)
    for t in range(4):
        assert_equal(len(tree[t]), t + 1)
    assert_almost_equal(tree[0][0], 0.1, atol=1e-12)
    assert_almost_equal(tree[1][0], 0.099, atol=1e-12)
    assert_almost_equal(tree[1][1], 0.102, atol=1e-12)
    assert_almost_equal(tree[2][0], 0.09801, atol=1e-12)
    assert_almost_equal(tree[2][1], 0.10098, atol=1e-12)
    assert_almost_equal(tree[2][2], 0.10404, atol=1e-12)


def test_value_of_cashflows() raises:
    # Book example, section 24.3: 3 period bond, coupon 10, face value 100, q=0.5
    var tree = interest_rate_trees_gbm_build(0.1, 1.02, 0.99, 3)
    var cashflows: List[Float64] = [0.0, 10.0, 10.0, 110.0]
    assert_almost_equal(interest_rate_trees_gbm_value_of_cashflows(cashflows, tree, 0.5), 98.5997, atol=1e-4)


def test_rate_lattice_of_callable_bond_example() raises:
    # Book example, section 24.4: r0=6%, u=1.2, d=0.9; lattice printed in percent
    var tree = interest_rate_trees_gbm_build(0.06, 1.2, 0.9, 10)
    assert_almost_equal(100.0 * tree[1][0], 5.4, atol=0.05)
    assert_almost_equal(100.0 * tree[1][1], 7.2, atol=0.05)
    assert_almost_equal(100.0 * tree[5][0], 3.5, atol=0.05)
    assert_almost_equal(100.0 * tree[5][5], 14.9, atol=0.05)
    assert_almost_equal(100.0 * tree[9][0], 2.3, atol=0.05)
    assert_almost_equal(100.0 * tree[9][4], 7.3, atol=0.05)
    assert_almost_equal(100.0 * tree[9][5], 9.8, atol=0.05)
    assert_almost_equal(100.0 * tree[9][9], 31.0, atol=0.05)


def test_straight_and_callable_bond() raises:
    # Book example, section 24.4: 10 year 6% bond, callable at 106 from year 6
    var tree = interest_rate_trees_gbm_build(0.06, 1.2, 0.9, 10)
    var cashflows = List[Float64]()
    cashflows.append(0.0)
    for _ in range(1, 10):
        cashflows.append(6.0)
    cashflows.append(106.0)
    assert_almost_equal(interest_rate_trees_gbm_value_of_cashflows(cashflows, tree, 0.5), 89.9017, atol=1e-4)
    assert_almost_equal(
        interest_rate_trees_gbm_value_of_callable_bond(cashflows, tree, 0.5, 6, 106.0), 89.2483, atol=1e-4
    )


def test_value_lattices_in_text() raises:
    # The lattices printed in the text of section 24.4 give B = 89.90 for the
    # straight bond and B = 89.35 for the callable bond, which is capped at
    # 106 from step 6 on. In C++ Code 24.3 the values of step t-1 are capped
    # when t >= first_call_time, so that lattice is first_call_time = 7; the
    # book's program passes 6 and prints 89.2483.
    var tree = interest_rate_trees_gbm_build(0.06, 1.2, 0.9, 10)
    var cashflows = List[Float64]()
    cashflows.append(0.0)
    for _ in range(1, 10):
        cashflows.append(6.0)
    cashflows.append(106.0)
    assert_almost_equal(interest_rate_trees_gbm_value_of_cashflows(cashflows, tree, 0.5), 89.90, atol=5e-3)
    assert_almost_equal(
        interest_rate_trees_gbm_value_of_callable_bond(cashflows, tree, 0.5, 7, 106.0), 89.35, atol=5e-3
    )


def test_bond_never_called_equals_straight_bond() raises:
    var tree = interest_rate_trees_gbm_build(0.06, 1.2, 0.9, 10)
    var cashflows = List[Float64]()
    cashflows.append(0.0)
    for _ in range(1, 10):
        cashflows.append(6.0)
    cashflows.append(106.0)
    var straight = interest_rate_trees_gbm_value_of_cashflows(cashflows, tree, 0.5)
    # a call price above every node value is never used
    assert_almost_equal(
        interest_rate_trees_gbm_value_of_callable_bond(cashflows, tree, 0.5, 6, 1000.0), straight, atol=1e-12
    )
    # nor is a call date after the last valuation date
    assert_almost_equal(
        interest_rate_trees_gbm_value_of_callable_bond(cashflows, tree, 0.5, 11, 106.0), straight, atol=1e-12
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
