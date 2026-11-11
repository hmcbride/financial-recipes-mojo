"""Chapter 20: pricing of bond options, basic models (C++ Code 20.1 - 20.3).

The bond price itself is assumed to follow a geometric Brownian motion, so
the Black Scholes formula and its binomial approximation apply with the
bond as the underlying security.
"""

from std.math import exp, log, pow, sqrt
from .normdist import N


def bond_option_price_put_zero_black_scholes(B: Float64, X: Float64, r: Float64, sigma: Float64, time: Float64) -> Float64:
    """Black Scholes price of a European put on a zero coupon bond (C++ Code 20.1).

    B: bond price, X: exercise price, r: interest rate, sigma: volatility of
    the bond price, time: time to maturity of the option.
    """
    var time_sqrt = sqrt(time)
    var d1 = (log(B / X) + r * time) / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt
    var d2 = d1 - (sigma * time_sqrt)
    var p = X * exp(-r * time) * N(-d2) - B * N(-d1)
    return p


def bond_option_price_put_coupon_bond_black_scholes(
    B: Float64,
    X: Float64,
    r: Float64,
    sigma: Float64,
    time: Float64,
    coupon_times: List[Float64],
    coupon_amounts: List[Float64],
) -> Float64:
    """Black Scholes price of a European put on a coupon bond (C++ Code 20.2).

    The bond price is reduced by the present value of the coupons paid
    before the option matures.
    """
    var adjusted_B = B
    for i in range(len(coupon_times)):
        if coupon_times[i] <= time:
            adjusted_B -= coupon_amounts[i] * exp(-r * coupon_times[i])
    return bond_option_price_put_zero_black_scholes(adjusted_B, X, r, sigma, time)


def bond_option_price_put_american_binomial(
    B: Float64, K: Float64, r: Float64, sigma: Float64, t: Float64, steps: Int
) -> Float64:
    """Binomial approximation to the price of an American put on a bond (C++ Code 20.3).

    B: bond price, K: exercise price, r: interest rate, sigma: volatility,
    t: time to maturity, steps: number of steps in the binomial tree.
    """
    var R = exp(r * (t / Float64(steps)))  # interest rate for each step
    var Rinv = 1.0 / R  # inverse of interest rate
    var u = exp(sigma * sqrt(t / Float64(steps)))  # up movement
    var uu = u * u
    var d = 1.0 / u
    var p_up = (R - d) / (u - d)
    var p_down = 1.0 - p_up
    var prices = List[Float64](length=steps + 1, fill=0.0)  # price of underlying
    var put_values = List[Float64](length=steps + 1, fill=0.0)  # value of corresponding put
    prices[0] = B * pow(d, Float64(steps))  # fill in the endnodes.
    for i in range(1, steps + 1):
        prices[i] = uu * prices[i - 1]
    for i in range(steps + 1):
        put_values[i] = max(0.0, K - prices[i])  # put payoffs at maturity
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            put_values[i] = (p_up * put_values[i + 1] + p_down * put_values[i]) * Rinv
            prices[i] = d * prices[i + 1]
            put_values[i] = max(put_values[i], K - prices[i])  # check for exercise
    return put_values[0]
