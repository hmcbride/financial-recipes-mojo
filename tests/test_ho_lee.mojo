from std.math import exp
from std.testing import TestSuite, assert_almost_equal, assert_equal
from finrecipes.term_structure import TermStructure, TermStructureFlat, bonds_price
from finrecipes.ho_lee import (
    hT,
    TermStructureHoLee,
    term_structure_ho_lee_build_term_structure_tree,
    TimeContingentCashFlows,
    build_time_series_of_bond_time_contingent_cash_flows,
    price_european_call_option_on_bond_using_ho_lee,
)


struct NelsonSiegel(Copyable, Movable, TermStructure):
    """The Nelson Siegel term structure of chapter 22 (C++ Code 22.1 and 22.2),
    which the book's Ho-Lee example uses as its second initial term structure."""

    var beta0: Float64
    var beta1: Float64
    var beta2: Float64
    var lam: Float64

    def __init__(out self, beta0: Float64, beta1: Float64, beta2: Float64, lam: Float64):
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.lam = lam

    def r(self, t: Float64) -> Float64:
        if t <= 0.0:
            return self.beta0
        var tl = t / self.lam
        return self.beta0 + (self.beta1 + self.beta2) * ((1.0 - exp(-tl)) / tl) + self.beta2 * exp(-tl)


def book_output[TS: TermStructure & Copyable & Deinitable](
    initial: TS, delta: Float64, pi: Float64, times: List[Float64], cflows: List[Float64], K: Float64, T: Int
) -> Float64:
    """What C++ Code 25.4 as printed returns when its read past the end of
    the vector of option values yields 0.5.

    The listing never computes the payoff in the top node of level T+1 and
    reads whatever follows the vector instead. That value reaches the price
    through the top node of every level, with weight pi and that node's one
    period discount factor at each step.
    """
    var stray: Float64 = 0.5
    var hl_tree = term_structure_ho_lee_build_term_structure_tree(initial, T + 1, delta, pi)
    var vec_cf = build_time_series_of_bond_time_contingent_cash_flows(times, cflows)
    var top_payoff = max(0.0, bonds_price(vec_cf[T + 1].times, vec_cf[T + 1].cash_flows, hl_tree[T + 1][T + 1]) - K)
    var weight: Float64 = 1.0
    for t in range(T + 1):
        weight *= pi * hl_tree[t][t].d(1.0)
    var price = price_european_call_option_on_bond_using_ho_lee(initial, delta, pi, times, cflows, K, Float64(T))
    return price - weight * (top_payoff - stray)


def test_hT() raises:
    assert_almost_equal(hT(0.0, 0.98, 0.5), 1.0, atol=1e-12)
    assert_almost_equal(hT(1.0, 0.98, 0.5), 1.0 / 0.99, atol=1e-12)
    assert_almost_equal(hT(2.0, 0.98, 0.5), 1.0 / (0.5 + 0.5 * 0.98 * 0.98), atol=1e-12)


def test_discount_factor() raises:
    # C++ Code 25.2 by hand in node (n=2, i=1) on a flat 10% initial term structure
    var initial = TermStructureFlat(0.1)
    var hl = TermStructureHoLee[TermStructureFlat](initial, 2, 1, 0.98, 0.5)
    var expected = exp(Float64(-0.1)) * hT(2.0, 0.98, 0.5) / hT(1.0, 0.98, 0.5) * hT(1.0, 0.98, 0.5) * 0.98
    assert_almost_equal(hl.d(1.0), expected, atol=1e-12)
    # the struct is a term structure: yields and forward rates follow from d
    assert_almost_equal(hl.r(1.0), 0.100204, atol=1e-6)
    assert_almost_equal(hl.d(0.0), 1.0, atol=1e-12)


def test_no_uncertainty_gives_initial_term_structure() raises:
    # with delta = 1 every node carries the forward term structure
    var initial = NelsonSiegel(0.09, 0.01, 0.01, 5.0)
    var hl = TermStructureHoLee[NelsonSiegel](initial, 3, 2, 1.0, 0.5)
    assert_almost_equal(hl.d(2.0), initial.d(5.0) / initial.d(3.0), atol=1e-12)


def test_tree_shape() raises:
    var initial = TermStructureFlat(0.1)
    var tree = term_structure_ho_lee_build_term_structure_tree(initial, 4, 0.98, 0.5)
    assert_equal(len(tree), 5)
    for t in range(5):
        assert_equal(len(tree[t]), t + 1)
        for j in range(t + 1):
            assert_equal(tree[t][j].n, t)
            assert_equal(tree[t][j].i, j)


def test_tree_is_arbitrage_free() raises:
    # from time 1 on, a T+1 period bond is worth the expected value of the
    # T period bond one step later, discounted with the one period bond
    var initial = NelsonSiegel(0.09, 0.01, 0.01, 5.0)
    var pi: Float64 = 0.5
    var tree = term_structure_ho_lee_build_term_structure_tree(initial, 4, 0.98, pi)
    for n in range(1, 4):
        for i in range(n + 1):
            for T in range(1, 4):
                var rolled_back = (
                    pi * tree[n + 1][i + 1].d(Float64(T)) + (1.0 - pi) * tree[n + 1][i].d(Float64(T))
                ) * tree[n][i].d(1.0)
                assert_almost_equal(tree[n][i].d(Float64(T + 1)), rolled_back, atol=1e-12)


def test_root_node_as_printed() raises:
    # C++ Code 25.2 applies h(T) at n = 0 as well, so the root node is not
    # the initial term structure itself
    var initial = NelsonSiegel(0.09, 0.01, 0.01, 5.0)
    var root = TermStructureHoLee[NelsonSiegel](initial, 0, 0, 0.98, 0.5)
    assert_almost_equal(root.d(1.0), initial.d(1.0) * hT(1.0, 0.98, 0.5), atol=1e-12)
    assert_almost_equal(root.d(3.0), initial.d(3.0) * hT(3.0, 0.98, 0.5), atol=1e-12)


def test_time_contingent_cash_flows() raises:
    var times: List[Float64] = [0.5, 1.0, 2.0]
    var cflows: List[Float64] = [5.0, 5.0, 105.0]
    var vec_cf = build_time_series_of_bond_time_contingent_cash_flows(times, cflows)
    assert_equal(len(vec_cf), 3)
    assert_equal(vec_cf[0].no_cflows(), 3)
    assert_equal(vec_cf[1].no_cflows(), 2)
    assert_almost_equal(vec_cf[1].times[0], 0.0, atol=1e-12)
    assert_almost_equal(vec_cf[1].times[1], 1.0, atol=1e-12)
    assert_almost_equal(vec_cf[1].cash_flows[1], 105.0, atol=1e-12)
    assert_equal(vec_cf[2].no_cflows(), 1)
    assert_almost_equal(vec_cf[2].times[0], 0.0, atol=1e-12)
    assert_almost_equal(vec_cf[2].cash_flows[0], 105.0, atol=1e-12)
    var one = TimeContingentCashFlows(times, cflows)
    assert_equal(one.no_cflows(), 3)


def test_european_call_on_bond() raises:
    # Book example, section 25.4: 5 year zero coupon bond, 3 year call, K=80,
    # delta=0.98, pi=0.5. The values are those of C++ Code 25.4 with the
    # payoff of the top node of the last level computed instead of read past
    # the end of the vector; see test_book_output for the printed numbers.
    var times: List[Float64] = [5.0]
    var cflows: List[Float64] = [100.0]
    var flat = TermStructureFlat(0.1)
    assert_almost_equal(
        price_european_call_option_on_bond_using_ho_lee(flat, 0.98, 0.5, times, cflows, 80.0, 3.0), 7.09845, atol=1e-5
    )
    var nelson_siegel = NelsonSiegel(0.09, 0.01, 0.01, 5.0)
    assert_almost_equal(
        price_european_call_option_on_bond_using_ho_lee(nelson_siegel, 0.98, 0.5, times, cflows, 80.0, 3.0),
        6.95282,
        atol=1e-5,
    )


def test_book_output() raises:
    # "Output from C++ program": c = 6.4857 (flat) and c = 6.35411 (Nelson Siegel).
    # Both follow from the port once the top node payoff is replaced by the
    # 0.5 that the listing's out of range read returned.
    var times: List[Float64] = [5.0]
    var cflows: List[Float64] = [100.0]
    var flat = TermStructureFlat(0.1)
    assert_almost_equal(book_output(flat, 0.98, 0.5, times, cflows, 80.0, 3), 6.4857, atol=1e-4)
    var nelson_siegel = NelsonSiegel(0.09, 0.01, 0.01, 5.0)
    assert_almost_equal(book_output(nelson_siegel, 0.98, 0.5, times, cflows, 80.0, 3), 6.35411, atol=1e-5)


def test_call_with_zero_exercise_price() raises:
    # with K = 0 and no uncertainty (delta = 1) the option is the bond itself
    var times: List[Float64] = [5.0]
    var cflows: List[Float64] = [100.0]
    var flat = TermStructureFlat(0.1)
    assert_almost_equal(
        price_european_call_option_on_bond_using_ho_lee(flat, 1.0, 0.5, times, cflows, 0.0, 3.0),
        100.0 * exp(Float64(-0.5)),
        atol=1e-10,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
