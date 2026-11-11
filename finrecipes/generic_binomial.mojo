"""Chapter 17: generic binomial pricing (C++ Code 17.1 - 17.5).

The terms of a derivative enter a binomial tree only through its value at
each node, so one routine can price any derivative once the payoff is passed
in as a function. C++ Code 17.1 repeats C++ Code 12.2
(`option_price_call_american_binomial` in `binomial.mojo`).
"""

from std.math import exp, pow, sqrt


def option_price_generic_binomial(
    S: Float64,
    K: Float64,
    generic_payoff: def(Float64, Float64) thin -> Float64,
    r: Float64,
    sigma: Float64,
    t: Float64,
    steps: Int,
) -> Float64:
    """Generic American binomial price (C++ Code 17.2).

    generic_payoff(S, K) gives the value of exercising at price S.
    """
    var R = exp(r * (t / Float64(steps)))  # interest rate for each step
    var Rinv = 1.0 / R  # inverse of interest rate
    var u = exp(sigma * sqrt(t / Float64(steps)))  # up movement
    var d = 1.0 / u
    var p_up = (R - d) / (u - d)
    var p_down = 1.0 - p_up
    var prices = List[Float64](length=steps + 1, fill=0.0)  # price of underlying
    prices[0] = S * pow(d, Float64(steps))  # fill in the endnodes.
    var uu = u * u
    for i in range(1, steps + 1):
        prices[i] = uu * prices[i - 1]
    var values = List[Float64](length=steps + 1, fill=0.0)  # value of corresponding derivative
    for i in range(steps + 1):
        values[i] = generic_payoff(prices[i], K)  # payoffs at maturity
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            values[i] = (p_up * values[i + 1] + p_down * values[i]) * Rinv  # value by not exercising
            prices[i] = d * prices[i + 1]
            values[i] = max(values[i], generic_payoff(prices[i], K))  # check for exercise
    return values[0]


def payoff_call(S: Float64, K: Float64) -> Float64:
    """Payoff of a call option (C++ Code 17.3)."""
    return max(0.0, S - K)


def payoff_put(S: Float64, K: Float64) -> Float64:
    """Payoff of a put option (C++ Code 17.3)."""
    return max(0.0, K - S)


def payoff_binary_call(S: Float64, K: Float64) -> Float64:
    """Payoff of a binary call: one dollar if S is at or above K (C++ Code 17.4)."""
    if S >= K:
        return 1.0
    return 0.0


def payoff_binary_put(S: Float64, K: Float64) -> Float64:
    """Payoff of a binary put: one dollar if S is at or below K (C++ Code 17.4)."""
    if S <= K:
        return 1.0
    return 0.0


def option_price_delta_generic_binomial(
    S: Float64,
    K: Float64,
    generic_payoff: def(Float64, Float64) thin -> Float64,
    r: Float64,
    sigma: Float64,
    t: Float64,
    no_steps: Int,
) -> Float64:
    """Generic binomial calculation of delta (C++ Code 17.5)."""
    var R = exp(r * (t / Float64(no_steps)))
    var Rinv = 1.0 / R
    var u = exp(sigma * sqrt(t / Float64(no_steps)))
    var d = 1.0 / u
    var uu = u * u
    var pUp = (R - d) / (u - d)
    var pDown = 1.0 - pUp
    var prices = List[Float64](length=no_steps + 1, fill=0.0)
    prices[0] = S * pow(d, Float64(no_steps))
    for i in range(1, no_steps + 1):
        prices[i] = uu * prices[i - 1]
    var values = List[Float64](length=no_steps + 1, fill=0.0)
    for i in range(no_steps + 1):
        values[i] = generic_payoff(prices[i], K)
    for CurrStep in range(no_steps - 1, 0, -1):
        for i in range(CurrStep + 1):
            prices[i] = d * prices[i + 1]
            values[i] = (pDown * values[i] + pUp * values[i + 1]) * Rinv
            values[i] = max(values[i], generic_payoff(prices[i], K))
    return (values[1] - values[0]) / (S * u - S * d)
