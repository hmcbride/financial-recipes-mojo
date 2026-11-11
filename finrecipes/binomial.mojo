"""Chapter 12: option pricing with binomial approximations (C++ Code 12.1 - 12.9).

All routines build a recombining tree with u = exp(sigma*sqrt(dt)), d = 1/u
and roll the option value backwards through it, keeping a single vector of
option values as the book does.
"""

from std.math import exp, pow, sqrt
from .black_scholes import Greeks


def option_price_call_european_binomial(
    S: Float64, K: Float64, r: Float64, sigma: Float64, t: Float64, steps: Int
) -> Float64:
    """European call, binomial approximation (C++ Code 12.1).

    S: spot price, K: exercise price, r: interest rate, sigma: volatility,
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
    prices[0] = S * pow(d, Float64(steps))  # fill in the endnodes.
    for i in range(1, steps + 1):
        prices[i] = uu * prices[i - 1]
    var call_values = List[Float64](length=steps + 1, fill=0.0)  # value of corresponding call
    for i in range(steps + 1):
        call_values[i] = max(0.0, prices[i] - K)  # call payoffs at maturity
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            call_values[i] = (p_up * call_values[i + 1] + p_down * call_values[i]) * Rinv
    return call_values[0]


def option_price_put_european_binomial(
    S: Float64, K: Float64, r: Float64, sigma: Float64, t: Float64, steps: Int
) -> Float64:
    """European put, binomial approximation (put version of C++ Code 12.1)."""
    var R = exp(r * (t / Float64(steps)))
    var Rinv = 1.0 / R
    var u = exp(sigma * sqrt(t / Float64(steps)))
    var uu = u * u
    var d = 1.0 / u
    var p_up = (R - d) / (u - d)
    var p_down = 1.0 - p_up
    var prices = List[Float64](length=steps + 1, fill=0.0)
    prices[0] = S * pow(d, Float64(steps))
    for i in range(1, steps + 1):
        prices[i] = uu * prices[i - 1]
    var put_values = List[Float64](length=steps + 1, fill=0.0)
    for i in range(steps + 1):
        put_values[i] = max(0.0, K - prices[i])  # put payoffs at maturity
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            put_values[i] = (p_up * put_values[i + 1] + p_down * put_values[i]) * Rinv
    return put_values[0]


def option_price_call_american_binomial(
    S: Float64, K: Float64, r: Float64, sigma: Float64, t: Float64, steps: Int
) -> Float64:
    """American call, binomial approximation (C++ Code 12.2, Matlab Code 12.1)."""
    var R = exp(r * (t / Float64(steps)))
    var Rinv = 1.0 / R
    var u = exp(sigma * sqrt(t / Float64(steps)))
    var d = 1.0 / u
    var p_up = (R - d) / (u - d)
    var p_down = 1.0 - p_up
    var prices = List[Float64](length=steps + 1, fill=0.0)  # price of underlying
    prices[0] = S * pow(d, Float64(steps))  # fill in the endnodes.
    var uu = u * u
    for i in range(1, steps + 1):
        prices[i] = uu * prices[i - 1]
    var call_values = List[Float64](length=steps + 1, fill=0.0)  # value of corresponding call
    for i in range(steps + 1):
        call_values[i] = max(0.0, prices[i] - K)  # call payoffs at maturity
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            call_values[i] = (p_up * call_values[i + 1] + p_down * call_values[i]) * Rinv
            prices[i] = d * prices[i + 1]
            call_values[i] = max(call_values[i], prices[i] - K)  # check for exercise
    return call_values[0]


def option_price_put_american_binomial(
    S: Float64, K: Float64, r: Float64, sigma: Float64, t: Float64, steps: Int
) -> Float64:
    """American put, binomial approximation (Matlab Code 12.2; put version of C++ Code 12.2)."""
    var R = exp(r * (t / Float64(steps)))
    var Rinv = 1.0 / R
    var u = exp(sigma * sqrt(t / Float64(steps)))
    var d = 1.0 / u
    var p_up = (R - d) / (u - d)
    var p_down = 1.0 - p_up
    var prices = List[Float64](length=steps + 1, fill=0.0)  # price of underlying
    prices[0] = S * pow(d, Float64(steps))  # fill in the endnodes.
    var uu = u * u
    for i in range(1, steps + 1):
        prices[i] = uu * prices[i - 1]
    var put_values = List[Float64](length=steps + 1, fill=0.0)  # value of corresponding put
    for i in range(steps + 1):
        put_values[i] = max(0.0, K - prices[i])  # put payoffs at maturity
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            put_values[i] = (p_up * put_values[i + 1] + p_down * put_values[i]) * Rinv
            prices[i] = d * prices[i + 1]
            put_values[i] = max(put_values[i], K - prices[i])  # check for exercise
    return put_values[0]


def option_price_delta_american_call_binomial(
    S: Float64, K: Float64, r: Float64, sigma: Float64, t: Float64, no_steps: Int
) -> Float64:
    """Delta of an American call, binomial approximation (C++ Code 12.3)."""
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
    var call_values = List[Float64](length=no_steps + 1, fill=0.0)
    for i in range(no_steps + 1):
        call_values[i] = max(0.0, prices[i] - K)
    for CurrStep in range(no_steps - 1, 0, -1):
        for i in range(CurrStep + 1):
            prices[i] = d * prices[i + 1]
            call_values[i] = (pDown * call_values[i] + pUp * call_values[i + 1]) * Rinv
            call_values[i] = max(call_values[i], prices[i] - K)  # check for exercise
    return (call_values[1] - call_values[0]) / (S * u - S * d)


def option_price_partials_american_call_binomial(
    S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64, no_steps: Int
) -> Greeks:
    """Hedge parameters of an American call, binomial approximation (C++ Code 12.4).

    The C++ routine returns delta, gamma, theta, vega and rho through
    reference arguments; here they come back as a `Greeks` struct.
    """
    var prices = List[Float64](length=no_steps + 1, fill=0.0)
    var call_values = List[Float64](length=no_steps + 1, fill=0.0)
    var delta_t = time / Float64(no_steps)
    var R = exp(r * delta_t)
    var Rinv = 1.0 / R
    var u = exp(sigma * sqrt(delta_t))
    var d = 1.0 / u
    var uu = u * u
    var pUp = (R - d) / (u - d)
    var pDown = 1.0 - pUp
    prices[0] = S * pow(d, Float64(no_steps))
    for i in range(1, no_steps + 1):
        prices[i] = uu * prices[i - 1]
    for i in range(no_steps + 1):
        call_values[i] = max(0.0, prices[i] - K)
    for CurrStep in range(no_steps - 1, 1, -1):
        for i in range(CurrStep + 1):
            prices[i] = d * prices[i + 1]
            call_values[i] = (pDown * call_values[i] + pUp * call_values[i + 1]) * Rinv
            call_values[i] = max(call_values[i], prices[i] - K)  # check for exercise
    var f22 = call_values[2]
    var f21 = call_values[1]
    var f20 = call_values[0]
    for i in range(2):
        prices[i] = d * prices[i + 1]
        call_values[i] = (pDown * call_values[i] + pUp * call_values[i + 1]) * Rinv
        call_values[i] = max(call_values[i], prices[i] - K)  # check for exercise
    var f11 = call_values[1]
    var f10 = call_values[0]
    prices[0] = d * prices[1]
    call_values[0] = (pDown * call_values[0] + pUp * call_values[1]) * Rinv
    call_values[0] = max(call_values[0], S - K)  # check for exercise on first date
    var f00 = call_values[0]
    var delta = (f11 - f10) / (S * u - S * d)
    var h = 0.5 * S * (uu - d * d)
    var gamma = ((f22 - f21) / (S * (uu - 1.0)) - (f21 - f20) / (S * (1.0 - d * d))) / h
    var theta = (f21 - f00) / (2.0 * delta_t)
    var diff: Float64 = 0.02
    var tmp_sigma = sigma + diff
    var tmp_prices = option_price_call_american_binomial(S, K, r, tmp_sigma, time, no_steps)
    var vega = (tmp_prices - f00) / diff
    diff = 0.05
    var tmp_r = r + diff
    tmp_prices = option_price_call_american_binomial(S, K, tmp_r, sigma, time, no_steps)
    var rho = (tmp_prices - f00) / diff
    return Greeks(delta, gamma, theta, vega, rho)


def option_price_call_american_binomial(
    S: Float64, K: Float64, r: Float64, y: Float64, sigma: Float64, t: Float64, steps: Int
) -> Float64:
    """American call on an underlying with continuous payout y (C++ Code 12.5).

    Overloads the six-argument routine, as in the book.
    """
    var R = exp(r * (t / Float64(steps)))  # interest rate for each step
    var Rinv = 1.0 / R  # inverse of interest rate
    var u = exp(sigma * sqrt(t / Float64(steps)))  # up movement
    var uu = u * u
    var d = 1.0 / u
    var p_up = (exp((r - y) * (t / Float64(steps))) - d) / (u - d)
    var p_down = 1.0 - p_up
    var prices = List[Float64](length=steps + 1, fill=0.0)  # price of underlying
    prices[0] = S * pow(d, Float64(steps))
    for i in range(1, steps + 1):
        prices[i] = uu * prices[i - 1]  # fill in the endnodes.
    var call_values = List[Float64](length=steps + 1, fill=0.0)  # value of corresponding call
    for i in range(steps + 1):
        call_values[i] = max(0.0, prices[i] - K)  # call payoffs at maturity
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            call_values[i] = (p_up * call_values[i + 1] + p_down * call_values[i]) * Rinv
            prices[i] = d * prices[i + 1]
            call_values[i] = max(call_values[i], prices[i] - K)  # check for exercise
    return call_values[0]


def option_price_call_american_proportional_dividends_binomial(
    S: Float64,
    K: Float64,
    r: Float64,
    sigma: Float64,
    time: Float64,
    no_steps: Int,
    dividend_times: List[Float64],
    dividend_yields: List[Float64],
) -> Float64:
    """American call on a stock paying proportional dividends (C++ Code 12.6).

    The last dividend date should be before the expiry date.
    """
    var no_dividends = len(dividend_times)
    if no_dividends == 0:
        return option_price_call_american_binomial(S, K, r, sigma, time, no_steps)  # price w/o dividends
    var delta_t = time / Float64(no_steps)
    var R = exp(r * delta_t)
    var Rinv = 1.0 / R
    var u = exp(sigma * sqrt(delta_t))
    var uu = u * u
    var d = 1.0 / u
    var pUp = (R - d) / (u - d)
    var pDown = 1.0 - pUp
    var dividend_steps = List[Int](length=no_dividends, fill=0)  # when dividends are paid
    for i in range(no_dividends):
        var step_of_dividend = dividend_times[i] / time * Float64(no_steps)
        dividend_steps[i] = Int(step_of_dividend)
    var prices = List[Float64](length=no_steps + 1, fill=0.0)
    var call_prices = List[Float64](length=no_steps + 1, fill=0.0)
    prices[0] = S * pow(d, Float64(no_steps))  # adjust downward terminal prices by dividends
    for i in range(no_dividends):
        prices[0] *= 1.0 - dividend_yields[i]
    for i in range(1, no_steps + 1):
        prices[i] = uu * prices[i - 1]
    for i in range(no_steps + 1):
        call_prices[i] = max(0.0, prices[i] - K)
    for step in range(no_steps - 1, -1, -1):
        for i in range(no_dividends):  # check whether dividend paid
            if step == dividend_steps[i]:
                for j in range(step + 2):
                    prices[j] *= 1.0 / (1.0 - dividend_yields[i])
        for i in range(step + 1):
            call_prices[i] = (pDown * call_prices[i] + pUp * call_prices[i + 1]) * Rinv
            prices[i] = d * prices[i + 1]
            call_prices[i] = max(call_prices[i], prices[i] - K)  # check for exercise
    return call_prices[0]


def option_price_call_american_discrete_dividends_binomial(
    S: Float64,
    K: Float64,
    r: Float64,
    sigma: Float64,
    t: Float64,
    steps: Int,
    dividend_times: List[Float64],
    dividend_amounts: List[Float64],
) -> Float64:
    """American call on a stock paying discrete dividends (C++ Code 12.7).

    The tree does not recombine after a dividend, so the routine builds a tree
    up to the first ex-dividend date and calls itself, with one dividend less,
    at each of that tree's terminal nodes.
    """
    var no_dividends = len(dividend_times)
    if no_dividends == 0:
        return option_price_call_american_binomial(S, K, r, sigma, t, steps)  # just do regular
    var steps_until_dividend = dividend_times[0] / t * Float64(steps)
    var steps_before_dividend = Int(steps_until_dividend)
    var R = exp(r * (t / Float64(steps)))
    var Rinv = 1.0 / R
    var u = exp(sigma * sqrt(t / Float64(steps)))
    var d = 1.0 / u
    var pUp = (R - d) / (u - d)
    var pDown = 1.0 - pUp
    var dividend_amount = dividend_amounts[0]
    var tmp_dividend_times = List[Float64](length=no_dividends - 1, fill=0.0)  # temporaries with
    var tmp_dividend_amounts = List[Float64](length=no_dividends - 1, fill=0.0)  # one less dividend
    for i in range(no_dividends - 1):
        tmp_dividend_amounts[i] = dividend_amounts[i + 1]
        tmp_dividend_times[i] = dividend_times[i + 1] - dividend_times[0]
    var prices = List[Float64](length=steps_before_dividend + 1, fill=0.0)
    var call_values = List[Float64](length=steps_before_dividend + 1, fill=0.0)
    prices[0] = S * pow(d, Float64(steps_before_dividend))
    for i in range(1, steps_before_dividend + 1):
        prices[i] = u * u * prices[i - 1]
    for i in range(steps_before_dividend + 1):
        var value_alive = option_price_call_american_discrete_dividends_binomial(
            prices[i] - dividend_amount,
            K,
            r,
            sigma,
            t - dividend_times[0],  # time after first dividend
            steps - steps_before_dividend,
            tmp_dividend_times,
            tmp_dividend_amounts,
        )
        call_values[i] = max(value_alive, prices[i] - K)  # compare to exercising now
    for step in range(steps_before_dividend - 1, -1, -1):
        for i in range(step + 1):
            prices[i] = d * prices[i + 1]
            call_values[i] = (pDown * call_values[i] + pUp * call_values[i + 1]) * Rinv
            call_values[i] = max(call_values[i], prices[i] - K)
    return call_values[0]


def futures_option_price_call_american_binomial(
    F: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64, no_steps: Int
) -> Float64:
    """American call on a futures contract (C++ Code 12.8).

    F: futures price, K: exercise price, r: interest rate, sigma: volatility,
    time: time to maturity, no_steps: number of steps.
    """
    var futures_prices = List[Float64](length=no_steps + 1, fill=0.0)
    var call_values = List[Float64](length=no_steps + 1, fill=0.0)
    var t_delta = time / Float64(no_steps)
    var Rinv = exp(-r * t_delta)
    var u = exp(sigma * sqrt(t_delta))
    var d = 1.0 / u
    var uu = u * u
    var pUp = (1.0 - d) / (u - d)  # note how probability is calculated
    var pDown = 1.0 - pUp
    futures_prices[0] = F * pow(d, Float64(no_steps))
    for i in range(1, no_steps + 1):
        futures_prices[i] = uu * futures_prices[i - 1]  # terminal tree nodes
    for i in range(no_steps + 1):
        call_values[i] = max(0.0, futures_prices[i] - K)
    for step in range(no_steps - 1, -1, -1):
        for i in range(step + 1):
            futures_prices[i] = d * futures_prices[i + 1]
            call_values[i] = (pDown * call_values[i] + pUp * call_values[i + 1]) * Rinv
            call_values[i] = max(call_values[i], futures_prices[i] - K)  # check for exercise
    return call_values[0]


def currency_option_price_call_american_binomial(
    S: Float64, K: Float64, r: Float64, r_f: Float64, sigma: Float64, time: Float64, no_steps: Int
) -> Float64:
    """American call on a currency (C++ Code 12.9).

    S: exchange rate, K: exercise price, r: domestic interest rate,
    r_f: foreign interest rate.
    """
    var exchange_rates = List[Float64](length=no_steps + 1, fill=0.0)
    var call_values = List[Float64](length=no_steps + 1, fill=0.0)
    var t_delta = time / Float64(no_steps)
    var Rinv = exp(-r * t_delta)
    var u = exp(sigma * sqrt(t_delta))
    var d = 1.0 / u
    var uu = u * u
    var pUp = (exp((r - r_f) * t_delta) - d) / (u - d)  # adjust for foreign int.rate
    var pDown = 1.0 - pUp
    exchange_rates[0] = S * pow(d, Float64(no_steps))
    for i in range(1, no_steps + 1):
        exchange_rates[i] = uu * exchange_rates[i - 1]  # terminal tree nodes
    for i in range(no_steps + 1):
        call_values[i] = max(0.0, exchange_rates[i] - K)
    for step in range(no_steps - 1, -1, -1):
        for i in range(step + 1):
            exchange_rates[i] = d * exchange_rates[i + 1]
            call_values[i] = (pDown * call_values[i] + pUp * call_values[i + 1]) * Rinv
            call_values[i] = max(call_values[i], exchange_rates[i] - K)  # check for exercise
    return call_values[0]
