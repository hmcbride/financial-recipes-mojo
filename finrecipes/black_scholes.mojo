"""Chapter 9: the Black Scholes formula (C++ Code 9.1 - 9.5, Exercise 9.1)."""

from std.math import exp, log, sqrt
from .normdist import n, N


def option_price_call_black_scholes(S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64) -> Float64:
    """European call price (C++ Code 9.1).

    S: spot price, K: strike, r: interest rate, sigma: volatility,
    time: time to maturity.
    """
    var time_sqrt = sqrt(time)
    var d1 = (log(S / K) + r * time) / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt
    var d2 = d1 - (sigma * time_sqrt)
    return S * N(d1) - K * exp(-r * time) * N(d2)


def option_price_put_black_scholes(S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64) -> Float64:
    """European put price (Exercise 9.1)."""
    var time_sqrt = sqrt(time)
    var d1 = (log(S / K) + r * time) / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt
    var d2 = d1 - (sigma * time_sqrt)
    return K * exp(-r * time) * N(-d2) - S * N(-d1)


def option_price_delta_call_black_scholes(S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64) -> Float64:
    """Delta of a European call (C++ Code 9.2)."""
    var time_sqrt = sqrt(time)
    var d1 = (log(S / K) + r * time) / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt
    return N(d1)


def option_price_delta_put_black_scholes(S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64) -> Float64:
    """Delta of a European put."""
    var time_sqrt = sqrt(time)
    var d1 = (log(S / K) + r * time) / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt
    return -N(-d1)


struct Greeks(Copyable, Movable):
    """Partial derivatives of an option price.

    The C++ routines return these through five reference arguments.
    """

    var delta: Float64  # partial wrt S
    var gamma: Float64  # second partial wrt S
    var theta: Float64  # partial wrt time
    var vega: Float64  # partial wrt sigma
    var rho: Float64  # partial wrt r

    def __init__(out self, delta: Float64, gamma: Float64, theta: Float64, vega: Float64, rho: Float64):
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.rho = rho


def option_price_partials_call_black_scholes(S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64) -> Greeks:
    """Partial derivatives of a European call (C++ Code 9.3)."""
    var time_sqrt = sqrt(time)
    var d1 = (log(S / K) + r * time) / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt
    var d2 = d1 - (sigma * time_sqrt)
    var delta = N(d1)
    var gamma = n(d1) / (S * sigma * time_sqrt)
    var theta = -(S * sigma * n(d1)) / (2.0 * time_sqrt) - r * K * exp(-r * time) * N(d2)
    var vega = S * time_sqrt * n(d1)
    var rho = K * time * exp(-r * time) * N(d2)
    return Greeks(delta, gamma, theta, vega, rho)


def option_price_partials_put_black_scholes(S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64) -> Greeks:
    """Partial derivatives of a European put."""
    var time_sqrt = sqrt(time)
    var d1 = (log(S / K) + r * time) / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt
    var d2 = d1 - (sigma * time_sqrt)
    var delta = -N(-d1)
    var gamma = n(d1) / (S * sigma * time_sqrt)
    var theta = -(S * sigma * n(d1)) / (2.0 * time_sqrt) + r * K * exp(-r * time) * N(-d2)
    var vega = S * time_sqrt * n(d1)
    var rho = -K * time * exp(-r * time) * N(-d2)
    return Greeks(delta, gamma, theta, vega, rho)


def option_price_implied_volatility_call_black_scholes_bisections(
    S: Float64, K: Float64, r: Float64, time: Float64, option_price: Float64
) -> Float64:
    """Implied volatility by bisection (C++ Code 9.4).

    Returns 0.0 if the price violates the arbitrage bound and -1e40 on failure.
    """
    if option_price < 0.99 * (S - K * exp(-time * r)):
        return 0.0  # option price is too low
    var ACCURACY: Float64 = 1.0e-5
    var MAX_ITERATIONS = 100
    var HIGH_VALUE: Float64 = 1e10
    var ERROR: Float64 = -1e40
    # bracket sigma: find a sigma with an estimated price above the actual price
    var sigma_low: Float64 = 1e-5
    var sigma_high: Float64 = 0.3
    var price = option_price_call_black_scholes(S, K, r, sigma_high, time)
    while price < option_price:
        sigma_high = 2.0 * sigma_high  # keep doubling
        price = option_price_call_black_scholes(S, K, r, sigma_high, time)
        if sigma_high > HIGH_VALUE:
            return ERROR  # panic, something wrong
    for _ in range(MAX_ITERATIONS):
        var sigma = (sigma_low + sigma_high) * 0.5
        price = option_price_call_black_scholes(S, K, r, sigma, time)
        var test = price - option_price
        if abs(test) < ACCURACY:
            return sigma
        if test < 0.0:
            sigma_low = sigma
        else:
            sigma_high = sigma
    return ERROR


def option_price_implied_volatility_call_black_scholes_newton(
    S: Float64, K: Float64, r: Float64, time: Float64, option_price: Float64
) -> Float64:
    """Implied volatility by Newton-Raphson (C++ Code 9.5).

    Returns 0.0 if the price violates the arbitrage bound and -99e10 on failure.
    """
    if option_price < 0.99 * (S - K * exp(-time * r)):
        return 0.0
    var MAX_ITERATIONS = 100
    var ACCURACY: Float64 = 1.0e-5
    var t_sqrt = sqrt(time)
    var sigma = (option_price / S) / (0.398 * t_sqrt)  # initial value
    for _ in range(MAX_ITERATIONS):
        var price = option_price_call_black_scholes(S, K, r, sigma, time)
        var diff = option_price - price
        if abs(diff) < ACCURACY:
            return sigma
        var d1 = (log(S / K) + r * time) / (sigma * t_sqrt) + 0.5 * sigma * t_sqrt
        var vega = S * t_sqrt * n(d1)
        sigma = sigma + diff / vega
    return -99e10
