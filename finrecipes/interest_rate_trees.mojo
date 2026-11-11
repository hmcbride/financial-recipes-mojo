"""Chapter 24: interest rate trees (C++ Code 24.1 - 24.3).

The short rate follows a geometric Brownian motion, approximated by a
recombining binomial tree. The tree is a `List[List[Float64]]`:
`tree[t][i]` is the rate at time t in node i, node 0 being the lowest rate.
"""

from std.math import exp


def interest_rate_trees_gbm_build(r0: Float64, u: Float64, d: Float64, n: Int) -> List[List[Float64]]:
    """Build an interest rate tree with n periods (C++ Code 24.1, Matlab Code 24.1)."""
    var tree = List[List[Float64]]()
    var r = List[Float64](length=1, fill=r0)
    tree.append(r.copy())
    for i in range(1, n + 1):
        var rtop = r[len(r) - 1] * u
        for j in range(i):
            r[j] = d * r[j]
        r.append(rtop)
        tree.append(r.copy())
    return tree^


def interest_rate_trees_gbm_value_of_cashflows(cflow: List[Float64], r_tree: List[List[Float64]], q: Float64) -> Float64:
    """Value a set of cash flows in an interest rate tree (C++ Code 24.2).

    cflow[t] is the cash flow at time t and q the probability weight on the
    lower-rate node of the next period.
    """
    var n = len(cflow)
    var values = List[List[Float64]]()
    for _ in range(n):
        values.append(List[Float64]())
    values[n - 1] = List[Float64](length=n, fill=cflow[n - 1])
    for t in range(n - 1, 0, -1):
        var value = List[Float64](length=t, fill=0.0)
        for i in range(t):
            value[i] = cflow[t - 1] + exp(-r_tree[t - 1][i]) * (q * values[t][i] + (1.0 - q) * values[t][i + 1])
        values[t - 1] = value^
    return values[0][0]


def interest_rate_trees_gbm_value_of_callable_bond(
    cflows: List[Float64], r_tree: List[List[Float64]], q: Float64, first_call_time: Int, call_price: Float64
) -> Float64:
    """Value a callable bond in an interest rate tree (C++ Code 24.3).

    From first_call_time on the issuer calls the bond whenever its value
    exceeds call_price.
    """
    var n = len(cflows)
    var values = List[List[Float64]]()
    for _ in range(n):
        values.append(List[Float64]())
    values[n - 1] = List[Float64](length=n, fill=cflows[n - 1])
    for t in range(n - 1, 0, -1):
        var value = List[Float64](length=t, fill=0.0)
        for i in range(t):
            value[i] = cflows[t - 1] + exp(-r_tree[t - 1][i]) * (q * values[t][i] + (1.0 - q) * values[t][i + 1])
            if t >= first_call_time:
                value[i] = min(value[i], call_price)
        values[t - 1] = value^
    return values[0][0]
