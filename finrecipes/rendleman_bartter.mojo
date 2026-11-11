"""Chapter 23: binomial term structure models (C++ Code 23.1).

The Rendleman and Bartter model applies the binomial approach of the Black
Scholes world to the short interest rate instead of to a price.
"""

from std.math import exp, pow, sqrt


def bond_option_price_call_zero_american_rendleman_bartter(
    X: Float64,
    option_maturity: Float64,
    S: Float64,
    M: Float64,
    interest: Float64,
    bond_maturity: Float64,
    maturity_payment: Float64,
    no_steps: Int,
) -> Float64:
    """Rendleman Bartter binomial price of a call on a zero coupon bond (C++ Code 23.1).

    X: exercise price, option_maturity: time to maturity of the option,
    S, M: term structure parameters (volatility and drift of the short rate),
    interest: current short interest rate, bond_maturity: time to maturity
    of the underlying bond, maturity_payment: payment of the bond at maturity,
    no_steps: number of steps in the tree, spanning the life of the bond.
    """
    var delta_t = bond_maturity / Float64(no_steps)
    var u = exp(S * sqrt(delta_t))
    var d = 1.0 / u
    var p_up = (exp(M * delta_t) - d) / (u - d)
    var p_down = 1.0 - p_up
    var r = List[Float64](length=no_steps + 1, fill=0.0)
    r[0] = interest * pow(d, Float64(no_steps))
    var uu = u * u
    for i in range(1, no_steps + 1):
        r[i] = r[i - 1] * uu
    var P = List[Float64](length=no_steps + 1, fill=0.0)
    for i in range(no_steps + 1):
        P[i] = maturity_payment
    var call_steps = Float64(no_steps) * option_maturity / bond_maturity
    var no_call_steps = Int(call_steps)
    for curr_step in range(no_steps, no_call_steps, -1):
        for i in range(curr_step):
            r[i] = r[i] * u
            P[i] = exp(-r[i] * delta_t) * (p_down * P[i] + p_up * P[i + 1])
    var C = List[Float64](length=no_call_steps + 1, fill=0.0)
    for i in range(no_call_steps + 1):
        C[i] = max(0.0, P[i] - X)
    for curr_step in range(no_call_steps, -1, -1):
        for i in range(curr_step):
            r[i] = r[i] * u
            P[i] = exp(-r[i] * delta_t) * (p_down * P[i] + p_up * P[i + 1])
            C[i] = max(P[i] - X, exp(-r[i] * delta_t) * (p_up * C[i + 1] + p_down * C[i]))
    return C[0]
