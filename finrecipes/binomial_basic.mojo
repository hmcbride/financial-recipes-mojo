"""Chapter 8: binomial option pricing (C++ Code 8.1 - 8.3, Exercise 8.3)."""

from std.math import exp, pow


def option_price_call_european_binomial_single_period(
    S: Float64, X: Float64, r: Float64, u: Float64, d: Float64
) -> Float64:
    """European call, one binomial period (C++ Code 8.1).

    S: spot price, X: exercise price, r: interest rate (per period),
    u: up movement, d: down movement.
    """
    var p_up = (exp(r) - d) / (u - d)
    var p_down = 1.0 - p_up
    var c_u = max(0.0, u * S - X)
    var c_d = max(0.0, d * S - X)
    var call_price = exp(-r) * (p_up * c_u + p_down * c_d)
    return call_price


def option_price_put_european_binomial_single_period(
    S: Float64, X: Float64, r: Float64, u: Float64, d: Float64
) -> Float64:
    """European put, one binomial period (Exercise 8.3)."""
    var p_up = (exp(r) - d) / (u - d)
    var p_down = 1.0 - p_up
    var p_u = max(0.0, X - u * S)
    var p_d = max(0.0, X - d * S)
    var put_price = exp(-r) * (p_up * p_u + p_down * p_d)
    return put_price


def binomial_tree(S0: Float64, u: Float64, d: Float64, no_steps: Int) -> List[List[Float64]]:
    """Build a binomial tree of prices of the underlying (C++ Code 8.2).

    Row i (counting from 0) holds the i + 1 possible prices after i steps,
    lowest first: tree[i][j] = S0 u^j d^(i-j).
    """
    var tree = List[List[Float64]]()
    for i in range(1, no_steps + 1):
        var S = List[Float64](length=i, fill=0.0)
        for j in range(i):
            S[j] = S0 * pow(u, Float64(j)) * pow(d, Float64(i - j - 1))
        tree.append(S.copy())
    return tree^


def option_price_call_european_binomial_multi_period_given_ud(
    S: Float64, K: Float64, r: Float64, u: Float64, d: Float64, no_periods: Int
) -> Float64:
    """European call, multiperiod binomial with given u and d (C++ Code 8.3).

    S: spot price, K: exercise price, r: interest rate (per period),
    u: up movement, d: down movement, no_periods: number of steps in the tree.

    As in the listing, the terminal prices are built by stepping up from
    S d^no_periods in multiples of u*u, which is the ratio u/d between
    neighbouring end nodes only when d = 1/u (the case of the book's
    example). For other choices of d the end nodes are not those of the tree.
    """
    var Rinv = exp(-r)  # inverse of interest rate
    var uu = u * u
    var p_up = (exp(r) - d) / (u - d)
    var p_down = 1.0 - p_up
    var prices = List[Float64](length=no_periods + 1, fill=0.0)  # price of underlying
    prices[0] = S * pow(d, Float64(no_periods))  # fill in the endnodes
    for i in range(1, no_periods + 1):
        prices[i] = uu * prices[i - 1]
    var call_values = List[Float64](length=no_periods + 1, fill=0.0)  # value of corresponding call
    for i in range(no_periods + 1):
        call_values[i] = max(0.0, prices[i] - K)  # call payoffs at maturity
    for step in range(no_periods - 1, -1, -1):
        for i in range(step + 1):
            call_values[i] = (p_up * call_values[i + 1] + p_down * call_values[i]) * Rinv
    return call_values[0]


def option_price_put_european_binomial_multi_period_given_ud(
    S: Float64, K: Float64, r: Float64, u: Float64, d: Float64, no_periods: Int
) -> Float64:
    """European put, multiperiod binomial with given u and d (Exercise 8.3).

    Built like C++ Code 8.3, so it also assumes d = 1/u.
    """
    var Rinv = exp(-r)  # inverse of interest rate
    var uu = u * u
    var p_up = (exp(r) - d) / (u - d)
    var p_down = 1.0 - p_up
    var prices = List[Float64](length=no_periods + 1, fill=0.0)  # price of underlying
    prices[0] = S * pow(d, Float64(no_periods))  # fill in the endnodes
    for i in range(1, no_periods + 1):
        prices[i] = uu * prices[i - 1]
    var put_values = List[Float64](length=no_periods + 1, fill=0.0)  # value of corresponding put
    for i in range(no_periods + 1):
        put_values[i] = max(0.0, K - prices[i])  # put payoffs at maturity
    for step in range(no_periods - 1, -1, -1):
        for i in range(step + 1):
            put_values[i] = (p_up * put_values[i + 1] + p_down * put_values[i]) * Rinv
    return put_values[0]
