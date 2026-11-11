"""Chapter 16: average, lookback and other exotic options (C++ Code 16.1 - 16.8).

The Monte Carlo routines take the payoff as a function pointer,
`def(List[Float64], Float64) thin -> Float64`: the simulated price path and
a constant (normally the exercise price). The two generic routines have the
same names as the terminal-price versions in `finrecipes.simulation`
(the book overloads them); import them from this module for path-dependent
payoffs.

The random numbers come from `finrecipes.random`, not the C library
`rand()` the book used, so simulated prices differ from the book's printed
output by sampling error. Call `random_seed` first for reproducible runs.
"""

from std.math import exp, log, pow, sqrt
from .black_scholes import option_price_call_black_scholes
from .normdist import N
from .random import random_normal
from .simulation import payoff_call


def option_price_put_bermudan_binomial(
    S: Float64,
    X: Float64,
    r: Float64,
    q: Float64,
    sigma: Float64,
    time: Float64,
    potential_exercise_times: List[Float64],
    steps: Int,
) -> Float64:
    """Binomial approximation to a Bermudan put option (C++ Code 16.1).

    q: continuous payout of the underlying. The option can be exercised at
    maturity and at the times in potential_exercise_times; early exercise is
    checked only at the tree steps corresponding to those times.
    """
    var delta_t = time / Float64(steps)
    var R = exp(r * delta_t)
    var Rinv = 1.0 / R
    var u = exp(sigma * sqrt(delta_t))
    var uu = u * u
    var d = 1.0 / u
    var p_up = (exp((r - q) * delta_t) - d) / (u - d)
    var p_down = 1.0 - p_up
    var prices = List[Float64](length=steps + 1, fill=0.0)
    var put_values = List[Float64](length=steps + 1, fill=0.0)
    var potential_exercise_steps = List[Int]()  # create list of steps at which exercise may happen
    for i in range(len(potential_exercise_times)):
        var t = potential_exercise_times[i]
        if t > 0.0 and t < time:
            var step_of_t = t / delta_t
            potential_exercise_steps.append(Int(step_of_t))
    prices[0] = S * pow(d, Float64(steps))  # fill in the endnodes.
    for i in range(1, steps + 1):
        prices[i] = uu * prices[i - 1]
    for i in range(steps + 1):
        put_values[i] = max(0.0, X - prices[i])  # put payoffs at maturity
    for step in range(steps - 1, -1, -1):
        var check_exercise_this_step = False
        for j in range(len(potential_exercise_steps)):
            if step == potential_exercise_steps[j]:
                check_exercise_this_step = True
        for i in range(step + 1):
            put_values[i] = (p_up * put_values[i + 1] + p_down * put_values[i]) * Rinv
            prices[i] = d * prices[i + 1]
            if check_exercise_this_step:
                put_values[i] = max(put_values[i], X - prices[i])
    return put_values[0]


def option_price_call_bermudan_binomial(
    S: Float64,
    X: Float64,
    r: Float64,
    q: Float64,
    sigma: Float64,
    time: Float64,
    potential_exercise_times: List[Float64],
    steps: Int,
) -> Float64:
    """Binomial approximation to a Bermudan call option.

    The call mirror of C++ Code 16.1 (the book lists only the put).
    """
    var delta_t = time / Float64(steps)
    var R = exp(r * delta_t)
    var Rinv = 1.0 / R
    var u = exp(sigma * sqrt(delta_t))
    var uu = u * u
    var d = 1.0 / u
    var p_up = (exp((r - q) * delta_t) - d) / (u - d)
    var p_down = 1.0 - p_up
    var prices = List[Float64](length=steps + 1, fill=0.0)
    var call_values = List[Float64](length=steps + 1, fill=0.0)
    var potential_exercise_steps = List[Int]()  # create list of steps at which exercise may happen
    for i in range(len(potential_exercise_times)):
        var t = potential_exercise_times[i]
        if t > 0.0 and t < time:
            var step_of_t = t / delta_t
            potential_exercise_steps.append(Int(step_of_t))
    prices[0] = S * pow(d, Float64(steps))  # fill in the endnodes.
    for i in range(1, steps + 1):
        prices[i] = uu * prices[i - 1]
    for i in range(steps + 1):
        call_values[i] = max(0.0, prices[i] - X)  # call payoffs at maturity
    for step in range(steps - 1, -1, -1):
        var check_exercise_this_step = False
        for j in range(len(potential_exercise_steps)):
            if step == potential_exercise_steps[j]:
                check_exercise_this_step = True
        for i in range(step + 1):
            call_values[i] = (p_up * call_values[i + 1] + p_down * call_values[i]) * Rinv
            prices[i] = d * prices[i + 1]
            if check_exercise_this_step:
                call_values[i] = max(call_values[i], prices[i] - X)
    return call_values[0]


def option_price_asian_geometric_average_price_call(
    S: Float64, K: Float64, r: Float64, q: Float64, sigma: Float64, time: Float64
) -> Float64:
    """Analytical price of an Asian geometric average price call (C++ Code 16.2).

    Kemna and Vorst (1990): the Black Scholes formula with volatility
    sigma/sqrt(3) and dividend yield (r + q + sigma^2/6)/2, for continuous
    sampling of the average.
    """
    var sigma_sqr = sigma * sigma
    var adj_div_yield = 0.5 * (r + q + sigma_sqr / 6.0)
    var adj_sigma = sigma / sqrt(3.0)
    var adj_sigma_sqr = adj_sigma * adj_sigma
    var time_sqrt = sqrt(time)
    var d1 = (log(S / K) + (r - adj_div_yield + 0.5 * adj_sigma_sqr) * time) / (adj_sigma * time_sqrt)
    var d2 = d1 - (adj_sigma * time_sqrt)
    var call_price = S * exp(-adj_div_yield * time) * N(d1) - K * exp(-r * time) * N(d2)
    return call_price


def option_price_european_lookback_call(
    S: Float64, Smin: Float64, r: Float64, q: Float64, sigma: Float64, time: Float64
) -> Float64:
    """Price of a European lookback call (C++ Code 16.3).

    Goldman, Sosin and Gatto (1979). Smin is the minimum price of the
    underlying observed so far. Returns 0 when r equals q, where the formula
    is not defined.
    """
    if r == q:
        return 0.0
    var sigma_sqr = sigma * sigma
    var time_sqrt = sqrt(time)
    var a1 = (log(S / Smin) + (r - q + sigma_sqr / 2.0) * time) / (sigma * time_sqrt)
    var a2 = a1 - sigma * time_sqrt
    var a3 = (log(S / Smin) + (-r + q + sigma_sqr / 2.0) * time) / (sigma * time_sqrt)
    var Y1 = 2.0 * (r - q - sigma_sqr / 2.0) * log(S / Smin) / sigma_sqr
    return (
        S * exp(-q * time) * N(a1)
        - S * exp(-q * time) * (sigma_sqr / (2.0 * (r - q))) * N(-a1)
        - Smin * exp(-r * time) * (N(a2) - (sigma_sqr / (2.0 * (r - q))) * exp(Y1) * N(-a3))
    )


def option_price_european_lookback_put(
    S: Float64, Smax: Float64, r: Float64, q: Float64, sigma: Float64, time: Float64
) -> Float64:
    """Price of a European lookback put.

    The book calls this routine in the example of section 16.5 without
    listing it. This is the put counterpart of C++ Code 16.3 (Goldman, Sosin
    and Gatto (1979), as given in Hull). Smax is the maximum price of the
    underlying observed so far. Returns 0 when r equals q, as the call does.
    """
    if r == q:
        return 0.0
    var sigma_sqr = sigma * sigma
    var time_sqrt = sqrt(time)
    var b1 = (log(Smax / S) + (-r + q + sigma_sqr / 2.0) * time) / (sigma * time_sqrt)
    var b2 = b1 - sigma * time_sqrt
    var b3 = (log(Smax / S) + (r - q - sigma_sqr / 2.0) * time) / (sigma * time_sqrt)
    var Y2 = 2.0 * (r - q - sigma_sqr / 2.0) * log(Smax / S) / sigma_sqr
    return (
        Smax * exp(-r * time) * (N(b1) - (sigma_sqr / (2.0 * (r - q))) * exp(Y2) * N(-b3))
        + S * exp(-q * time) * (sigma_sqr / (2.0 * (r - q))) * N(-b2)
        - S * exp(-q * time) * N(b2)
    )


def simulate_lognormally_distributed_sequence(
    S: Float64, r: Float64, sigma: Float64, time: Float64, no_steps: Int
) -> List[Float64]:
    """Simulate a sequence of lognormally distributed variables (C++ Code 16.4).

    time: time to final date, no_steps: number of steps. Returns the
    no_steps prices after the current one; the last is the terminal price.
    """
    var prices = List[Float64](length=no_steps, fill=0.0)
    var delta_t = time / Float64(no_steps)
    var R = (r - 0.5 * sigma * sigma) * delta_t
    var SD = sigma * sqrt(delta_t)
    var S_t = S  # initialize at current price
    for i in range(no_steps):
        S_t = S_t * exp(R + SD * random_normal())
        prices[i] = S_t
    return prices^


def derivative_price_simulate_european_option_generic(
    S: Float64,
    K: Float64,
    r: Float64,
    sigma: Float64,
    time: Float64,
    payoff: def(List[Float64], Float64) thin -> Float64,
    no_steps: Int,
    no_sims: Int,
) -> Float64:
    """Generic routine for pricing European options on the price path (C++ Code 16.5).

    payoff(prices, X) gives the value at maturity as a function of the
    simulated price path and the constant K.
    """
    var sum_payoffs: Float64 = 0.0
    for _ in range(no_sims):
        var prices = simulate_lognormally_distributed_sequence(S, r, sigma, time, no_steps)
        sum_payoffs += payoff(prices, K)
    return exp(-r * time) * (sum_payoffs / Float64(no_sims))


def payoff_arithmetric_average_call(prices: List[Float64], K: Float64) -> Float64:
    """Payoff of an Asian call on the arithmetic average price (C++ Code 16.6)."""
    var sum: Float64 = 0.0
    for i in range(len(prices)):
        sum += prices[i]
    var avg = sum / Float64(len(prices))
    return max(0.0, avg - K)


def payoff_geometric_average_call(prices: List[Float64], K: Float64) -> Float64:
    """Payoff of an Asian call on the geometric average price (C++ Code 16.6)."""
    var logsum = log(prices[0])
    for i in range(1, len(prices)):
        logsum += log(prices[i])
    var avg = exp(logsum / Float64(len(prices)))
    return max(0.0, avg - K)


def payoff_lookback_call(prices: List[Float64], unused_variable: Float64) -> Float64:
    """Payoff of a lookback call: terminal price less the minimum (C++ Code 16.7)."""
    var m = prices[0]
    for i in range(1, len(prices)):
        if prices[i] < m:
            m = prices[i]
    return prices[len(prices) - 1] - m  # always positive or zero


def payoff_lookback_put(prices: List[Float64], unused_variable: Float64) -> Float64:
    """Payoff of a lookback put: the maximum less the terminal price (C++ Code 16.7)."""
    var m = prices[0]
    for i in range(1, len(prices)):
        if prices[i] > m:
            m = prices[i]
    return m - prices[len(prices) - 1]  # max is always larger or equal.


def derivative_price_simulate_european_option_generic_with_control_variate(
    S: Float64,
    K: Float64,
    r: Float64,
    sigma: Float64,
    time: Float64,
    payoff: def(List[Float64], Float64) thin -> Float64,
    no_steps: Int,
    no_sims: Int,
) -> Float64:
    """Generic path-dependent pricing with a control variate (C++ Code 16.8).

    The control variate is an at the money European call, priced both by
    Black Scholes and from the terminal prices of the same simulated paths.
    """
    var c_bs = option_price_call_black_scholes(S, S, r, sigma, time)  # price an at the money Black Scholes call
    var sum_payoffs: Float64 = 0.0
    var sum_payoffs_bs: Float64 = 0.0
    for _ in range(no_sims):
        var prices = simulate_lognormally_distributed_sequence(S, r, sigma, time, no_steps)
        var S1 = prices[len(prices) - 1]
        sum_payoffs += payoff(prices, K)
        sum_payoffs_bs += payoff_call(S1, S)  # simulate at the money Black Scholes price
    var c_sim = exp(-r * time) * (sum_payoffs / Float64(no_sims))
    var c_bs_sim = exp(-r * time) * (sum_payoffs_bs / Float64(no_sims))
    c_sim += c_bs - c_bs_sim
    return c_sim
