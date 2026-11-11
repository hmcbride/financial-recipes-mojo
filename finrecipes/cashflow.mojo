"""Chapter 3: the value of time (C++ Code 3.1 - 3.4).

Present values with discrete (annual) and continuous compounding, and the
internal rate of return.
"""

from std.math import exp, pow


def cash_flow_pv_discrete(cflow_times: List[Float64], cflow_amounts: List[Float64], r: Float64) -> Float64:
    """Present value with discrete, annual compounding (C++ Code 3.1)."""
    var PV: Float64 = 0.0
    for t in range(len(cflow_times)):
        PV += cflow_amounts[t] / pow(1.0 + r, cflow_times[t])
    return PV


def cash_flow_irr_discrete(cflow_times: List[Float64], cflow_amounts: List[Float64]) -> Float64:
    """Internal rate of return by bisection (C++ Code 3.2).

    Simple minded irr function, adapted from a routine in Numerical Recipes
    in C. Will find one root (if it exists). Returns -1e30 on error.
    """
    var ERROR: Float64 = -1e30
    if len(cflow_times) != len(cflow_amounts):
        return ERROR
    var ACCURACY: Float64 = 1.0e-5
    var MAX_ITERATIONS = 50
    var x1: Float64 = 0.0
    var x2: Float64 = 0.2
    # create an initial bracket, with a root somewhere between bot, top
    var f1 = cash_flow_pv_discrete(cflow_times, cflow_amounts, x1)
    var f2 = cash_flow_pv_discrete(cflow_times, cflow_amounts, x2)
    for _ in range(MAX_ITERATIONS):
        if (f1 * f2) < 0.0:
            break
        if abs(f1) < abs(f2):
            x1 += 1.6 * (x1 - x2)
            f1 = cash_flow_pv_discrete(cflow_times, cflow_amounts, x1)
        else:
            x2 += 1.6 * (x2 - x1)
            f2 = cash_flow_pv_discrete(cflow_times, cflow_amounts, x2)
    if f2 * f1 > 0.0:
        return ERROR
    var f = cash_flow_pv_discrete(cflow_times, cflow_amounts, x1)
    var rtb: Float64
    var dx: Float64
    if f < 0.0:
        rtb = x1
        dx = x2 - x1
    else:
        rtb = x2
        dx = x1 - x2
    for _ in range(MAX_ITERATIONS):
        dx *= 0.5
        var x_mid = rtb + dx
        var f_mid = cash_flow_pv_discrete(cflow_times, cflow_amounts, x_mid)
        if f_mid <= 0.0:
            rtb = x_mid
        if abs(f_mid) < ACCURACY or abs(dx) < ACCURACY:
            return x_mid
    return ERROR  # error.


def sgn(r: Float64) -> Int:
    """Sign of r: 1 for r >= 0, otherwise -1 (C++ Code 3.3)."""
    if r >= 0.0:
        return 1
    return -1


def cash_flow_unique_irr(cflow_times: List[Float64], cflow_amounts: List[Float64]) -> Bool:
    """Test for uniqueness of the IRR (C++ Code 3.3).

    A necessary condition only: the IRR may be well defined even if this
    returns False. First Descartes' rule (count the sign changes in the cash
    flows), then the sign changes in the aggregated cash flows (Norstrom).
    """
    var sign_changes = 0  # first check Descartes rule
    for t in range(1, len(cflow_times)):
        if sgn(cflow_amounts[t - 1]) != sgn(cflow_amounts[t]):
            sign_changes += 1
    if sign_changes == 0:
        return False  # can not find any irr
    if sign_changes == 1:
        return True
    var A = cflow_amounts[0]  # check the aggregate cash flows, due to Norstrom
    sign_changes = 0
    for t in range(1, len(cflow_times)):
        var sign_before = sgn(A)
        A += cflow_amounts[t]
        if sign_before != sgn(A):
            sign_changes += 1
    if sign_changes <= 1:
        return True
    return False


def cash_flow_pv(cflow_times: List[Float64], cflow_amounts: List[Float64], r: Float64) -> Float64:
    """Present value with continuously compounded interest (C++ Code 3.4)."""
    var PV: Float64 = 0.0
    for t in range(len(cflow_times)):
        PV += cflow_amounts[t] * exp(-r * cflow_times[t])
    return PV
