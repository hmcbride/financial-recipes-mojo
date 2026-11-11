"""Chapter 11: extending the Black Scholes formula (C++ Code 11.1 - 11.6).

The put versions are the formulas the chapter states in Exercises 11.1,
11.3, 11.4 and 11.5.
"""

from std.math import exp, log, sqrt, pow
from .normdist import N, N2
from .black_scholes import option_price_call_black_scholes, option_price_put_black_scholes


def option_price_european_call_payout(
    S: Float64, X: Float64, r: Float64, q: Float64, sigma: Float64, time: Float64
) -> Float64:
    """European call, continuous payout from the underlying (C++ Code 11.1).

    S: spot price, X: strike (exercise) price, r: interest rate,
    q: yield on underlying, sigma: volatility, time: time to maturity.
    """
    var sigma_sqr = sigma * sigma
    var time_sqrt = sqrt(time)
    var d1 = (log(S / X) + (r - q + 0.5 * sigma_sqr) * time) / (sigma * time_sqrt)
    var d2 = d1 - (sigma * time_sqrt)
    var call_price = S * exp(-q * time) * N(d1) - X * exp(-r * time) * N(d2)
    return call_price


def option_price_european_put_payout(
    S: Float64, X: Float64, r: Float64, q: Float64, sigma: Float64, time: Float64
) -> Float64:
    """European put, continuous payout from the underlying (Exercise 11.1)."""
    var sigma_sqr = sigma * sigma
    var time_sqrt = sqrt(time)
    var d1 = (log(S / X) + (r - q + 0.5 * sigma_sqr) * time) / (sigma * time_sqrt)
    var d2 = d1 - (sigma * time_sqrt)
    var put_price = X * exp(-r * time) * N(-d2) - S * exp(-q * time) * N(-d1)
    return put_price


def option_price_european_call_dividends(
    S: Float64,
    K: Float64,
    r: Float64,
    sigma: Float64,
    time_to_maturity: Float64,
    dividend_times: List[Float64],
    dividend_amounts: List[Float64],
) -> Float64:
    """European call on a stock paying known discrete dividends (C++ Code 11.2).

    The present value of the dividends paid before maturity is subtracted
    from the stock price, which then goes into the Black Scholes formula.
    """
    var adjusted_S = S
    for i in range(len(dividend_times)):
        if dividend_times[i] <= time_to_maturity:
            adjusted_S -= dividend_amounts[i] * exp(-r * dividend_times[i])
    return option_price_call_black_scholes(adjusted_S, K, r, sigma, time_to_maturity)


def option_price_european_put_dividends(
    S: Float64,
    K: Float64,
    r: Float64,
    sigma: Float64,
    time_to_maturity: Float64,
    dividend_times: List[Float64],
    dividend_amounts: List[Float64],
) -> Float64:
    """European put on a stock paying known discrete dividends.

    The put counterpart of C++ Code 11.2: the same adjustment of the stock price.
    """
    var adjusted_S = S
    for i in range(len(dividend_times)):
        if dividend_times[i] <= time_to_maturity:
            adjusted_S -= dividend_amounts[i] * exp(-r * dividend_times[i])
    return option_price_put_black_scholes(adjusted_S, K, r, sigma, time_to_maturity)


def option_price_american_call_one_dividend(
    S: Float64, K: Float64, r: Float64, sigma: Float64, tau: Float64, D1: Float64, tau1: Float64
) -> Float64:
    """American call on a stock paying one dividend, Roll-Geske-Whaley (C++ Code 11.3).

    S: stock price, K: exercise price, r: interest rate, sigma: volatility,
    tau: time to maturity, D1: dividend amount, tau1: time to the dividend.
    """
    if D1 <= K * (1.0 - exp(-r * (tau - tau1))):  # check for no exercise
        return option_price_call_black_scholes(S - exp(-r * tau1) * D1, K, r, sigma, tau)
    var ACCURACY: Float64 = 1e-6  # decrease this for more accuracy
    var sigma_sqr = sigma * sigma
    var tau_sqrt = sqrt(tau)
    var tau1_sqrt = sqrt(tau1)
    var rho = -sqrt(tau1 / tau)
    # first find the S_bar that solves c = S_bar + D1 - K
    var S_low: Float64 = 0.0  # the simplest: binomial search
    var S_high = S  # start by finding a very high S above S_bar
    var c = option_price_call_black_scholes(S_high, K, r, sigma, tau - tau1)
    var test = c - S_high - D1 + K
    while test > 0.0 and S_high <= 1e10:
        S_high *= 2.0
        c = option_price_call_black_scholes(S_high, K, r, sigma, tau - tau1)
        test = c - S_high - D1 + K
    if S_high > 1e10:  # early exercise never optimal, find BS value
        return option_price_call_black_scholes(S - D1 * exp(-r * tau1), K, r, sigma, tau)
    var S_bar = 0.5 * S_high  # now find S_bar that solves c = S_bar - D + K
    c = option_price_call_black_scholes(S_bar, K, r, sigma, tau - tau1)
    test = c - S_bar - D1 + K
    while abs(test) > ACCURACY and (S_high - S_low) > ACCURACY:
        if test < 0.0:
            S_high = S_bar
        else:
            S_low = S_bar
        S_bar = 0.5 * (S_high + S_low)
        c = option_price_call_black_scholes(S_bar, K, r, sigma, tau - tau1)
        test = c - S_bar - D1 + K
    var a1 = (log((S - D1 * exp(-r * tau1)) / K) + (r + 0.5 * sigma_sqr) * tau) / (sigma * tau_sqrt)
    var a2 = a1 - sigma * tau_sqrt
    var b1 = (log((S - D1 * exp(-r * tau1)) / S_bar) + (r + 0.5 * sigma_sqr) * tau1) / (sigma * tau1_sqrt)
    var b2 = b1 - sigma * tau1_sqrt
    var C = (
        (S - D1 * exp(-r * tau1)) * N(b1)
        + (S - D1 * exp(-r * tau1)) * N2(a1, -b1, rho)
        - (K * exp(-r * tau)) * N2(a2, -b2, rho)
        - (K - D1) * exp(-r * tau1) * N(b2)
    )
    return C


def futures_option_price_call_european_black(
    F: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64
) -> Float64:
    """European call on a futures contract, Black's model (C++ Code 11.4).

    F: futures price, K: exercise price, r: interest rate,
    sigma: volatility, time: time to maturity.
    """
    var sigma_sqr = sigma * sigma
    var time_sqrt = sqrt(time)
    var d1 = (log(F / K) + 0.5 * sigma_sqr * time) / (sigma * time_sqrt)
    var d2 = d1 - sigma * time_sqrt
    return exp(-r * time) * (F * N(d1) - K * N(d2))


def futures_option_price_put_european_black(
    F: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64
) -> Float64:
    """European put on a futures contract, Black's model (Exercise 11.3).

    This is the routine the example program of section 11.3 calls.
    """
    var sigma_sqr = sigma * sigma
    var time_sqrt = sqrt(time)
    var d1 = (log(F / K) + 0.5 * sigma_sqr * time) / (sigma * time_sqrt)
    var d2 = d1 - sigma * time_sqrt
    return exp(-r * time) * (K * N(-d2) - F * N(-d1))


def currency_option_price_call_european(
    S: Float64, X: Float64, r: Float64, r_f: Float64, sigma: Float64, time: Float64
) -> Float64:
    """European call on a currency (C++ Code 11.5).

    S: exchange rate, X: exercise price, r: domestic interest rate,
    r_f: foreign interest rate, sigma: volatility, time: time to maturity.
    """
    var sigma_sqr = sigma * sigma
    var time_sqrt = sqrt(time)
    var d1 = (log(S / X) + (r - r_f + (0.5 * sigma_sqr)) * time) / (sigma * time_sqrt)
    var d2 = d1 - sigma * time_sqrt
    return S * exp(-r_f * time) * N(d1) - X * exp(-r * time) * N(d2)


def currency_option_price_put_european(
    S: Float64, X: Float64, r: Float64, r_f: Float64, sigma: Float64, time: Float64
) -> Float64:
    """European put on a currency (Exercise 11.4)."""
    var sigma_sqr = sigma * sigma
    var time_sqrt = sqrt(time)
    var d1 = (log(S / X) + (r - r_f + (0.5 * sigma_sqr)) * time) / (sigma * time_sqrt)
    var d2 = d1 - sigma * time_sqrt
    return X * exp(-r * time) * N(-d2) - S * exp(-r_f * time) * N(-d1)


def option_price_american_perpetual_call(S: Float64, K: Float64, r: Float64, q: Float64, sigma: Float64) -> Float64:
    """American perpetual call (C++ Code 11.6).

    S: price of the underlying, K: exercise price, r: interest rate,
    q: dividend yield, sigma: volatility.
    """
    var sigma_sqr = sigma * sigma
    var h1 = 0.5 - ((r - q) / sigma_sqr)
    h1 += sqrt(pow(((r - q) / sigma_sqr - 0.5), 2.0) + 2.0 * r / sigma_sqr)
    var pric = (K / (h1 - 1.0)) * pow(((h1 - 1.0) / h1) * (S / K), h1)
    return pric


def option_price_american_perpetual_put(S: Float64, K: Float64, r: Float64, q: Float64, sigma: Float64) -> Float64:
    """American perpetual put (Exercise 11.5)."""
    var sigma_sqr = sigma * sigma
    var h2 = 0.5 - ((r - q) / sigma_sqr)
    h2 -= sqrt(pow(((r - q) / sigma_sqr - 0.5), 2.0) + 2.0 * r / sigma_sqr)
    var pric = (K / (1.0 - h2)) * pow(((h2 - 1.0) / h2) * (S / K), h2)
    return pric
