"""Chapter 25: building term structure trees using the Ho and Lee (1986)
approach (Header file 25.1, C++ Code 25.1 - 25.4).

The C++ class `term_structure_class_ho_lee` holds a pointer to the initial
term structure. Here `TermStructureHoLee` is a struct generic over the type
of the initial term structure, which can be any `TermStructure`.

Two deviations from the printed listings, both needed for defined behaviour:

- C++ Code 25.3 builds a tree with a hard-coded 5 levels and ignores its
  `no_steps` argument. The port builds levels 0 to no_steps, which is the
  same tree for the book's example (no_steps = 4).
- C++ Code 25.4 fills only T+1 of the T+2 option values at the last level of
  the tree and then reads the missing one, past the end of the vector. The
  port fills all T+2 values. See
  `price_european_call_option_on_bond_using_ho_lee` for the effect on the
  book's printed output.
"""

from std.math import pow
from .term_structure import TermStructure, bonds_price

comptime EPSILON: Float64 = 0.000001


def hT(T: Float64, delta: Float64, pi: Float64) -> Float64:
    """The perturbation function h(T) of Ho and Lee (C++ Code 25.2)."""
    return 1.0 / (pi + (1.0 - pi) * pow(delta, T))


struct TermStructureHoLee[TS: TermStructure & Copyable & Deinitable](Copyable, Movable, TermStructure):
    """Term structure in node i at time n of a Ho-Lee tree (Header file 25.1, C++ Code 25.1).

    initial_term: the term structure fitted at time 0, n: time (number of
    steps into the tree), i: node at that time, delta and pi: the Ho-Lee
    parameters.
    """

    var initial_term: Self.TS
    var n: Int
    var i: Int
    var delta: Float64
    var pi: Float64

    def __init__(out self, fitted_term: Self.TS, n: Int, i: Int, delta: Float64, pi: Float64):
        self.initial_term = fitted_term.copy()
        self.n = n
        self.i = i
        self.delta = delta
        self.pi = pi

    def d(self, T: Float64) -> Float64:
        """Discount factor for a payment T periods after this node (C++ Code 25.2).

        As printed, the listing multiplies with h(T) at every level, also at
        n = 0, so the root node gives h(T) times the discount factor of the
        initial term structure. From n = 1 on the tree is arbitrage free.
        """
        var time_n = Float64(self.n)
        var d = self.initial_term.d(T + time_n) / self.initial_term.d(time_n)
        for j in range(1, self.n):
            var time_n_j = Float64(self.n - j)
            d *= hT(T + time_n_j, self.delta, self.pi) / hT(time_n_j, self.delta, self.pi)
        d *= hT(T, self.delta, self.pi) * pow(self.delta, T * Float64(self.n - self.i))
        return d


def term_structure_ho_lee_build_term_structure_tree[TS: TermStructure & Copyable & Deinitable](
    initial: TS, no_steps: Int, delta: Float64, pi: Float64
) -> List[List[TermStructureHoLee[TS]]]:
    """Build a tree of Ho-Lee term structures (C++ Code 25.3).

    tree[t][j] is the term structure at time t in node j, for t = 0..no_steps.
    The printed listing loops over a hard-coded 5 levels whatever no_steps is.
    """
    var hl_tree = List[List[TermStructureHoLee[TS]]]()
    for t in range(no_steps + 1):
        var level = List[TermStructureHoLee[TS]]()
        for j in range(t + 1):
            level.append(TermStructureHoLee[TS](initial, t, j, delta, pi))
        hl_tree.append(level^)
    return hl_tree^


struct TimeContingentCashFlows(Copyable, Movable):
    """The cash flows of a bond that remain at some date (C++ Code 25.4)."""

    var times: List[Float64]
    var cash_flows: List[Float64]

    def __init__(out self, in_times: List[Float64], in_cflows: List[Float64]):
        self.times = in_times.copy()
        self.cash_flows = in_cflows.copy()

    def no_cflows(self) -> Int:
        return len(self.times)


def build_time_series_of_bond_time_contingent_cash_flows(
    initial_times: List[Float64], initial_cflows: List[Float64]
) -> List[TimeContingentCashFlows]:
    """The remaining cash flows of a bond at dates 0, 1, 2, ... (C++ Code 25.4).

    Entry t holds the cash flows still to come at date t, with their times
    measured from date t.
    """
    var vec_cf = List[TimeContingentCashFlows]()
    var times = initial_times.copy()
    var cflows = initial_cflows.copy()
    while len(times) > 0:
        vec_cf.append(TimeContingentCashFlows(times, cflows))
        var tmp_times = List[Float64]()
        var tmp_cflows = List[Float64]()
        for i in range(len(times)):
            if times[i] - 1.0 >= 0.0:
                tmp_times.append(times[i] - 1.0)
                tmp_cflows.append(cflows[i])
        times = tmp_times^
        cflows = tmp_cflows^
    return vec_cf^


def price_european_call_option_on_bond_using_ho_lee[TS: TermStructure & Copyable & Deinitable](
    initial: TS,
    delta: Float64,
    pi: Float64,
    underlying_bond_cflow_times: List[Float64],
    underlying_bond_cflows: List[Float64],
    K: Float64,
    time_to_maturity: Float64,
) -> Float64:
    """European call on a straight bond in a Ho-Lee tree (C++ Code 25.4).

    As in the listing, the option payoff is taken at level T+1 of the tree,
    T being the whole number of periods in time_to_maturity, and rolled back
    to level 0.

    The listing computes the payoff in nodes 0..T of level T+1 only, and the
    roll back then reads the value of node T+1 past the end of the vector.
    Here all T+2 payoffs are computed. The numbers printed in the book
    (6.4857 and 6.35411) are what the listing gives when that stray read
    returns 0.5; with the payoff of node T+1 in its place the prices are
    7.09845 and 6.95282.
    """
    var T = Int(time_to_maturity + EPSILON)
    var hl_tree = term_structure_ho_lee_build_term_structure_tree(initial, T + 1, delta, pi)
    var vec_cf = build_time_series_of_bond_time_contingent_cash_flows(
        underlying_bond_cflow_times, underlying_bond_cflows
    )
    var values = List[Float64](length=T + 2, fill=0.0)
    for i in range(T + 2):
        values[i] = max(0.0, bonds_price(vec_cf[T + 1].times, vec_cf[T + 1].cash_flows, hl_tree[T + 1][i]) - K)
    for t in range(T, -1, -1):
        var values_this = List[Float64](length=t + 1, fill=0.0)
        for i in range(t + 1):
            values_this[i] = (pi * values[i + 1] + (1.0 - pi) * values[i]) * hl_tree[t][i].d(1.0)
        values = values_this^
    return values[0]
