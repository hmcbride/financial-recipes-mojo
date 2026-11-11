"""Chapter 10: warrants (C++ Code 10.1)."""

from std.math import log, sqrt
from .normdist import N
from .black_scholes import option_price_call_black_scholes

comptime EPSILON: Float64 = 0.00001


def warrant_price_adjusted_black_scholes(
    S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64, m: Float64, n: Float64
) -> Float64:
    """Adjusted Black Scholes value for a warrant (C++ Code 10.1).

    S: stock price, K: exercise price, r: interest rate, sigma: volatility,
    time: time to maturity, m: number of warrants outstanding,
    n: number of shares outstanding.

    Solves W = n/(n+m) C_BS(S + (m/n) W, K, r, sigma, time) for W by Newton
    iteration on g(W) = W - n/(n+m) C_BS(S + (m/n) W, ...).

    The slope used in the iteration is copied from the listing, which
    evaluates d1 at S + m/n and uses 1 - (m/n) N(d1). The text of section
    10.2 instead gives g'(W) = 1 - m/(m+n) N(d1) with d1 evaluated at
    S + (m/n) W. The slope only affects the path of the iteration: the loop
    stops when |g(W)| <= EPSILON, so the returned value solves the warrant
    equation either way.
    """
    var time_sqrt = sqrt(time)
    var w = (n / (n + m)) * option_price_call_black_scholes(S, K, r, sigma, time)
    var g = w - (n / (n + m)) * option_price_call_black_scholes(S + (m / n) * w, K, r, sigma, time)
    while abs(g) > EPSILON:
        var d1 = (log((S + (m / n)) / K) + r * time) / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt
        var gprime = 1.0 - (m / n) * N(d1)
        w = w - g / gprime
        g = w - (n / (n + m)) * option_price_call_black_scholes(S + (m / n) * w, K, r, sigma, time)
    return w
