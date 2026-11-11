"""Chapter 4: bond pricing with a flat term structure (C++ Code 4.1 - 4.10).

The `_discrete` routines use discrete, annual compounding; the others use
continuously compounded interest.

`bonds_price`, `bonds_duration` and `bonds_convexity` here take the interest
rate as a number. The C++ overloads that take a term structure object
(chapter 5) are in `term_structure.mojo` under the same names.
"""

from std.math import exp, pow


# Discrete, annual compounding


def bonds_price_discrete(times: List[Float64], cashflows: List[Float64], r: Float64) -> Float64:
    """Bond price with discrete, annual compounding (C++ Code 4.1)."""
    var p: Float64 = 0.0
    for i in range(len(times)):
        p += cashflows[i] / pow(1.0 + r, times[i])
    return p


def bonds_yield_to_maturity_discrete(times: List[Float64], cashflows: List[Float64], bondprice: Float64) -> Float64:
    """Bond yield to maturity with discrete, annual compounding, by bisection (C++ Code 4.2)."""
    var ACCURACY: Float64 = 1e-5
    var MAX_ITERATIONS = 200
    var bot: Float64 = 0.0
    var top: Float64 = 1.0
    while bonds_price_discrete(times, cashflows, top) > bondprice:
        top = top * 2.0
    var r = 0.5 * (top + bot)
    for _ in range(MAX_ITERATIONS):
        var diff = bonds_price_discrete(times, cashflows, r) - bondprice
        if abs(diff) < ACCURACY:
            return r
        if diff > 0.0:
            bot = r
        else:
            top = r
        r = 0.5 * (top + bot)
    return r


def bonds_duration_discrete(times: List[Float64], cashflows: List[Float64], r: Float64) -> Float64:
    """Bond duration with discrete, annual compounding and a flat term structure (C++ Code 4.3)."""
    var B: Float64 = 0.0
    var D: Float64 = 0.0
    for i in range(len(times)):
        D += times[i] * cashflows[i] / pow(1.0 + r, times[i])
        B += cashflows[i] / pow(1.0 + r, times[i])
    return D / B


def bonds_duration_macaulay_discrete(times: List[Float64], cashflows: List[Float64], bond_price: Float64) -> Float64:
    """Macaulay duration of a bond (C++ Code 4.4)."""
    var y = bonds_yield_to_maturity_discrete(times, cashflows, bond_price)
    return bonds_duration_discrete(times, cashflows, y)  # use YTM in duration calculation


def bonds_duration_modified_discrete(times: List[Float64], cashflows: List[Float64], bond_price: Float64) -> Float64:
    """Modified duration (C++ Code 4.5)."""
    var y = bonds_yield_to_maturity_discrete(times, cashflows, bond_price)
    var D = bonds_duration_discrete(times, cashflows, y)
    return D / (1.0 + y)


def bonds_convexity_discrete(times: List[Float64], cashflows: List[Float64], r: Float64) -> Float64:
    """Bond convexity with a flat term structure and annual compounding (C++ Code 4.6)."""
    var Cx: Float64 = 0.0
    for i in range(len(times)):
        Cx += cashflows[i] * times[i] * (times[i] + 1.0) / pow(1.0 + r, times[i])
    var B = bonds_price_discrete(times, cashflows, r)
    return (Cx / ((1.0 + r) * (1.0 + r))) / B


# Continuously compounded interest


def bonds_price(cashflow_times: List[Float64], cashflows: List[Float64], r: Float64) -> Float64:
    """Bond price with continuously compounded interest and a flat term structure (C++ Code 4.7)."""
    var p: Float64 = 0.0
    for i in range(len(cashflow_times)):
        p += exp(-r * cashflow_times[i]) * cashflows[i]
    return p


def bonds_duration(cashflow_times: List[Float64], cashflows: List[Float64], r: Float64) -> Float64:
    """Bond duration with continuously compounded interest and a flat term structure (C++ Code 4.8)."""
    var S: Float64 = 0.0
    var D1: Float64 = 0.0
    for i in range(len(cashflow_times)):
        S += cashflows[i] * exp(-r * cashflow_times[i])
        D1 += cashflow_times[i] * cashflows[i] * exp(-r * cashflow_times[i])
    return D1 / S


def bonds_yield_to_maturity(cashflow_times: List[Float64], cashflows: List[Float64], bondprice: Float64) -> Float64:
    """Bond yield to maturity with continuously compounded interest, by bisection.

    Called by C++ Code 4.9. The chapter does not print this routine; it is
    the bisection of C++ Code 4.2 with `bonds_price` in place of
    `bonds_price_discrete`.
    """
    var ACCURACY: Float64 = 1e-5
    var MAX_ITERATIONS = 200
    var bot: Float64 = 0.0
    var top: Float64 = 1.0
    while bonds_price(cashflow_times, cashflows, top) > bondprice:
        top = top * 2.0
    var r = 0.5 * (top + bot)
    for _ in range(MAX_ITERATIONS):
        var diff = bonds_price(cashflow_times, cashflows, r) - bondprice
        if abs(diff) < ACCURACY:
            return r
        if diff > 0.0:
            bot = r
        else:
            top = r
        r = 0.5 * (top + bot)
    return r


def bonds_duration_macaulay(cashflow_times: List[Float64], cashflows: List[Float64], bond_price: Float64) -> Float64:
    """Macaulay duration with continuously compounded interest and a flat term structure (C++ Code 4.9)."""
    var y = bonds_yield_to_maturity(cashflow_times, cashflows, bond_price)
    return bonds_duration(cashflow_times, cashflows, y)  # use YTM in duration


def bonds_convexity(times: List[Float64], cashflows: List[Float64], r: Float64) -> Float64:
    """Bond convexity with continuously compounded interest and a flat term structure (C++ Code 4.10)."""
    var C: Float64 = 0.0
    for i in range(len(times)):
        C += cashflows[i] * times[i] * times[i] * exp(-r * times[i])
    var B = bonds_price(times, cashflows, r)
    return C / B
