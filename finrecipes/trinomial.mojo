"""Chapter 18: trinomial trees (C++ Code 18.1)."""

from std.math import exp, sqrt


def option_price_put_american_trinomial(
    S: Float64, K: Float64, r: Float64, q: Float64, sigma: Float64, t: Float64, steps: Int
) -> Float64:
    """American put priced in a trinomial tree (C++ Code 18.1, Matlab Code 18.1).

    S: spot price, K: exercise price, r: interest rate, q: continuous payout
    of the underlying, sigma: volatility, t: time to maturity,
    steps: number of steps in the tree.
    """
    var delta_t = t / Float64(steps)
    var Rinv = exp(-r * delta_t)
    var sigma_sqr = sigma * sigma
    var u = exp(sigma * sqrt(3.0 * delta_t))
    var d = 1.0 / u
    var p_u = 1.0 / 6.0 + sqrt(delta_t / (12.0 * sigma_sqr)) * (r - q - 0.5 * sigma_sqr)
    var p_m: Float64 = 2.0 / 3.0
    var p_d = 1.0 / 6.0 - sqrt(delta_t / (12.0 * sigma_sqr)) * (r - q - 0.5 * sigma_sqr)
    var Stree = List[List[Float64]]()  # price of underlying in a tree
    var Svec = List[Float64]()
    Svec.append(S)
    for _ in range(1, steps + 1):
        Stree.append(Svec.copy())
        Svec.insert(0, Svec[0] * d)  # use the fact that only the extreme values change.
        Svec.append(Svec[len(Svec) - 1] * u)
    var m = len(Svec)
    var values_next = List[Float64](length=m, fill=0.0)  # value of option next step
    for i in range(m):
        values_next[i] = max(0.0, K - Svec[i])  # put payoffs at maturity
    for step in range(steps - 1, -1, -1):
        m = len(Stree[step])
        var values = List[Float64](length=m, fill=0.0)  # value of option
        for i in range(m):
            values[i] = (p_u * values_next[i + 2] + p_m * values_next[i + 1] + p_d * values_next[i]) * Rinv
            values[i] = max(values[i], K - Stree[step][i])  # check for exercise
        values_next = values^
    return values_next[0]
