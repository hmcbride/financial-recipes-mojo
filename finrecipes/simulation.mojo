"""Chapter 14: option pricing by simulation (C++ Code 14.1 - 14.8).

The random numbers come from `finrecipes.random`, not the C library
`rand()` the book used, so simulated prices differ from the book's printed
output by sampling error. Call `random_seed` first for reproducible runs.

The book's "function prototypes" become function-pointer arguments:
a payoff is any `def(Float64, Float64) thin -> Float64` taking the terminal
price of the underlying and a constant (normally the exercise price).
"""

from std.math import exp, sqrt
from .black_scholes import option_price_call_black_scholes
from .random import random_normal


def simulate_lognormal_random_variable(S: Float64, r: Float64, sigma: Float64, time: Float64) -> Float64:
    """Simulate a lognormally distributed random variable (C++ Code 14.1).

    S: current value of variable, r: interest rate, sigma: volatility,
    time: time to final date.
    """
    var R = (r - 0.5 * sigma * sigma) * time
    var SD = sigma * sqrt(time)
    return S * exp(R + SD * random_normal())


def option_price_call_european_simulated(
    S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64, no_sims: Int
) -> Float64:
    """European call option priced by simulation (C++ Code 14.2)."""
    var R = (r - 0.5 * sigma * sigma) * time
    var SD = sigma * sqrt(time)
    var sum_payoffs: Float64 = 0.0
    for _ in range(no_sims):
        var S_T = S * exp(R + SD * random_normal())
        sum_payoffs += max(0.0, S_T - K)
    return exp(-r * time) * (sum_payoffs / Float64(no_sims))


def option_price_put_european_simulated(
    S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64, no_sims: Int
) -> Float64:
    """European put option priced by simulation.

    Used in the example following C++ Code 14.2; the put mirror of that listing.
    """
    var R = (r - 0.5 * sigma * sigma) * time
    var SD = sigma * sqrt(time)
    var sum_payoffs: Float64 = 0.0
    for _ in range(no_sims):
        var S_T = S * exp(R + SD * random_normal())
        sum_payoffs += max(0.0, K - S_T)
    return exp(-r * time) * (sum_payoffs / Float64(no_sims))


def option_price_delta_call_european_simulated(
    S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64, no_sims: Int
) -> Float64:
    """Delta of a European call estimated by Monte Carlo (C++ Code 14.3).

    The two option prices, at S and S+q, use the same sequence of random
    numbers.
    """
    var R = (r - 0.5 * sigma * sigma) * time
    var SD = sigma * sqrt(time)
    var sum_payoffs: Float64 = 0.0
    var sum_payoffs_q: Float64 = 0.0
    var q = S * 0.01
    for _ in range(no_sims):
        var Z = random_normal()
        var S_T = S * exp(R + SD * Z)
        sum_payoffs += max(0.0, S_T - K)
        var S_T_q = (S + q) * exp(R + SD * Z)
        sum_payoffs_q += max(0.0, S_T_q - K)
    var c = exp(-r * time) * (sum_payoffs / Float64(no_sims))
    var c_q = exp(-r * time) * (sum_payoffs_q / Float64(no_sims))
    return (c_q - c) / q


def option_price_delta_put_european_simulated(
    S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64, no_sims: Int
) -> Float64:
    """Delta of a European put estimated by Monte Carlo.

    Used in the example following C++ Code 14.3; the put mirror of that listing.
    """
    var R = (r - 0.5 * sigma * sigma) * time
    var SD = sigma * sqrt(time)
    var sum_payoffs: Float64 = 0.0
    var sum_payoffs_q: Float64 = 0.0
    var q = S * 0.01
    for _ in range(no_sims):
        var Z = random_normal()
        var S_T = S * exp(R + SD * Z)
        sum_payoffs += max(0.0, K - S_T)
        var S_T_q = (S + q) * exp(R + SD * Z)
        sum_payoffs_q += max(0.0, K - S_T_q)
    var p = exp(-r * time) * (sum_payoffs / Float64(no_sims))
    var p_q = exp(-r * time) * (sum_payoffs_q / Float64(no_sims))
    return (p_q - p) / q


def payoff_call(S: Float64, K: Float64) -> Float64:
    """Payoff of a call option at maturity (C++ Code 14.4)."""
    return max(0.0, S - K)


def payoff_put(S: Float64, K: Float64) -> Float64:
    """Payoff of a put option at maturity (C++ Code 14.4)."""
    return max(0.0, K - S)


def derivative_price_simulate_european_option_generic(
    S: Float64,
    K: Float64,
    r: Float64,
    sigma: Float64,
    time: Float64,
    payoff: def(Float64, Float64) thin -> Float64,
    no_sims: Int,
) -> Float64:
    """Generic simulation pricing of a European derivative (C++ Code 14.5).

    payoff(price, X) gives the value at maturity as a function of the
    terminal price of the underlying and the constant K.
    """
    var sum_payoffs: Float64 = 0.0
    for _ in range(no_sims):
        var S_T = simulate_lognormal_random_variable(S, r, sigma, time)
        sum_payoffs += payoff(S_T, K)
    return exp(-r * time) * (sum_payoffs / Float64(no_sims))


def derivative_price_simulate_european_option_generic_with_control_variate(
    S: Float64,
    X: Float64,
    r: Float64,
    sigma: Float64,
    time: Float64,
    payoff: def(Float64, Float64) thin -> Float64,
    no_sims: Int,
) -> Float64:
    """Generic simulation pricing with a control variate (C++ Code 14.6).

    The control variate is an at the money European call, priced both by
    Black Scholes and with the same simulated terminal values.
    """
    var c_bs = option_price_call_black_scholes(S, S, r, sigma, time)  # price an at the money Black Scholes call
    var sum_payoffs: Float64 = 0.0
    var sum_payoffs_bs: Float64 = 0.0
    for _ in range(no_sims):
        var S_T = simulate_lognormal_random_variable(S, r, sigma, time)
        sum_payoffs += payoff(S_T, X)
        sum_payoffs_bs += payoff_call(S_T, S)  # simulate at the money Black Scholes price
    var c_sim = exp(-r * time) * (sum_payoffs / Float64(no_sims))
    var c_bs_sim = exp(-r * time) * (sum_payoffs_bs / Float64(no_sims))
    c_sim += c_bs - c_bs_sim
    return c_sim


def derivative_price_simulate_european_option_generic_with_antithetic_variate(
    S: Float64,
    K: Float64,
    r: Float64,
    sigma: Float64,
    time: Float64,
    payoff: def(Float64, Float64) thin -> Float64,
    no_sims: Int,
) -> Float64:
    """Generic simulation pricing with antithetic variates (C++ Code 14.7).

    Each simulated unit normal x is used twice, as x and as -x.
    """
    var R = (r - 0.5 * sigma * sigma) * time
    var SD = sigma * sqrt(time)
    var sum_payoffs: Float64 = 0.0
    for _ in range(no_sims):
        var x = random_normal()
        var S1 = S * exp(R + SD * x)
        sum_payoffs += payoff(S1, K)
        var S2 = S * exp(R + SD * (-x))
        sum_payoffs += payoff(S2, K)
    return exp(-r * time) * (sum_payoffs / Float64(2 * no_sims))


def payoff_cash_or_nothing_call(S: Float64, K: Float64) -> Float64:
    """Payoff of a cash or nothing call paying 1 (C++ Code 14.8)."""
    if S >= K:
        return 1.0
    return 0.0


def payoff_asset_or_nothing_call(S: Float64, K: Float64) -> Float64:
    """Payoff of an asset or nothing call (C++ Code 14.8)."""
    if S >= K:
        return S
    return 0.0
