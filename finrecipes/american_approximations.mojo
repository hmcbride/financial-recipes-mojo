"""Chapter 15: pricing American options, approximations (C++ Code 15.1 - 15.5).

Each listing is ported as printed, under the book's routine name. Four of
the five listings do not compute what the chapter's formulas (and the
papers they come from) say, so for those a second routine with the suffix
`_corrected` implements the formula. The differences, found by comparing
the listings with the formulas and with binomial prices:

- C++ Code 15.1 (Johnson put): correct.
- C++ Code 15.2 (Geske-Johnson put): the critical prices are computed for
  the wrong times to maturity and from European instead of Bermudan values,
  two discount factors are interchanged, and d1 and d2 are interchanged in
  the trivariate terms.
- C++ Code 15.3 (Barone-Adesi Whaley call): b is the cost of carry in the
  formulas, but it is passed to `option_price_european_call_payout`, which
  expects the payout rate q = r - b. With the book's example the Newton
  iteration for the critical price does not converge and the seed value is
  used. The printed output 5.74339 is reproduced by the port as listed.
- C++ Code 15.4 (Bjerksund-Stensland call): the same mix up of b and q in
  the European lower bound, and h(T) uses K*K where the 1993 paper has B0,
  which makes the result depend on the units of the prices.
- C++ Code 15.5 (Bjerksund-Stensland put): the transformation is
  P(S, X, T, r, b) = C(X, S, T, r - b, -b); the listing passes +b.
"""

from std.math import exp, log, pow, sqrt
from .black_scholes import option_price_put_black_scholes, option_price_delta_put_black_scholes
from .black_scholes_extensions import option_price_european_call_payout as _option_price_european_call_payout
from .normdist import n, N, N2
from .normdist3 import N3

comptime ACCURACY: Float64 = 1e-6


def _max(a: Float64, b: Float64) -> Float64:
    """The C++ std::max, including its behaviour when an argument is not a number."""
    if a < b:
        return b
    return a


def option_price_american_put_approximated_johnson(
    S: Float64, X: Float64, r: Float64, sigma: Float64, time: Float64
) -> Float64:
    """The Johnson (1983) approximation to an American put price (C++ Code 15.1)."""
    var sigma_sqr = sigma * sigma
    var a0: Float64 = 3.9649
    var a1: Float64 = 0.032325
    var b0: Float64 = 1.040803
    var b1: Float64 = 0.00963
    var gamma = 2.0 * r / sigma_sqr
    var m = (sigma_sqr * time) / (b0 * sigma_sqr * time + b1)
    var Sc = X * pow(gamma / (1.0 + gamma), m)
    var l = log(S / Sc) / log(X / Sc)
    var alpha = pow((r * time) / (a0 * r * time + a1), l)
    var P = alpha * option_price_put_black_scholes(S, X * exp(r * time), r, sigma, time) + (
        1.0 - alpha
    ) * option_price_put_black_scholes(S, X, r, sigma, time)
    var p = option_price_put_black_scholes(S, X, r, sigma, time)  # for safety use the Black Scholes as lower bound
    return _max(p, P)


def _d1(S: Float64, X: Float64, r: Float64, sigma: Float64, tau: Float64) -> Float64:
    return (log(S / X) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt(tau))


def _d2(S: Float64, X: Float64, r: Float64, sigma: Float64, tau: Float64) -> Float64:
    return _d1(S, X, r, sigma, tau) - sigma * sqrt(tau)


def _calc_P2(
    S: Float64,
    X: Float64,
    r: Float64,
    sigma: Float64,
    time: Float64,
    t2: Float64,
    S2_bar: Float64,
    rho12: Float64,
) -> Float64:
    """Put with two exercise dates, t2 and time (`calcP2` of C++ Code 15.2).

    S2_bar is the critical price at t2, below which the put is exercised.
    """
    var P2 = X * exp(-r * t2) * N(-_d2(S, S2_bar, r, sigma, t2))
    P2 -= S * N(-_d1(S, S2_bar, r, sigma, t2))
    P2 += X * exp(-r * time) * N2(_d2(S, S2_bar, r, sigma, t2), -_d2(S, X, r, sigma, time), -rho12)
    P2 -= S * N2(_d1(S, S2_bar, r, sigma, t2), -_d1(S, X, r, sigma, time), -rho12)
    return P2


def option_price_american_put_approximated_geske_johnson(
    S: Float64, X: Float64, r: Float64, sigma: Float64, time: Float64
) -> Float64:
    """Geske Johnson approximation of an American put, as listed (C++ Code 15.2).

    The listing departs from the formulas of section 15.2, see the module
    docstring; its P3 is often below P2 (the result is then the average of
    the European and the two-date Bermudan price) and it overprices puts
    that are in the money. Use
    `option_price_american_put_approximated_geske_johnson_corrected` for
    the approximation of Geske and Johnson (1984).
    """
    var P1 = option_price_put_black_scholes(S, X, r, sigma, time)
    var rho12 = 1.0 / sqrt(2.0)
    var rho13 = 1.0 / sqrt(3.0)
    var rho23 = sqrt(2.0 / 3.0)
    var t2 = time / 2.0
    var t23 = time * 2.0 / 3.0
    var t3 = time / 3.0
    var Si = S
    var S2_bar = S
    var g: Float64 = 1.0
    while abs(g) > ACCURACY:
        g = Si - X + option_price_put_black_scholes(Si, X, r, sigma, t2)
        var gprime = 1.0 + option_price_delta_put_black_scholes(Si, X, r, sigma, t2)
        S2_bar = Si
        Si = Si - g / gprime
    var P2 = _calc_P2(S, X, r, sigma, time, t2, S2_bar, rho12)
    P2 = _max(P1, P2)  # for safety, use one less step as lower bound
    var S23_bar = S2_bar
    g = 1.0
    while abs(g) > ACCURACY:
        g = Si - X + option_price_put_black_scholes(Si, X, r, sigma, t23)
        var gprime = 1.0 + option_price_delta_put_black_scholes(Si, X, r, sigma, t23)
        S23_bar = Si
        Si = Si - g / gprime
    var S3_bar = S23_bar
    g = 1.0
    while abs(g) > ACCURACY:
        g = Si - X + option_price_put_black_scholes(Si, X, r, sigma, t3)
        var gprime = 1.0 + option_price_delta_put_black_scholes(Si, X, r, sigma, t3)
        S3_bar = Si
        Si = Si - g / gprime
    var P3 = X * exp(-r * t3) * N(-_d2(S, S3_bar, r, sigma, t3))
    P3 -= S * N(-_d1(S, S3_bar, r, sigma, t3))
    P3 += X * exp(-r * time) * N2(_d2(S, S3_bar, r, sigma, t3), -_d2(S, S23_bar, r, sigma, t23), -rho12)
    P3 -= S * N2(_d1(S, S3_bar, r, sigma, t3), -_d1(S, S23_bar, r, sigma, t23), -rho12)
    P3 += (
        X
        * exp(-r * t23)
        * N3(
            _d1(S, S3_bar, r, sigma, t3),
            _d1(S, S23_bar, r, sigma, t23),
            -_d1(S, X, r, sigma, time),
            rho12,
            -rho13,
            -rho23,
        )
    )
    P3 -= S * N3(
        _d2(S, S3_bar, r, sigma, t3),
        _d2(S, S23_bar, r, sigma, t23),
        -_d2(S, X, r, sigma, time),
        rho12,
        -rho13,
        -rho23,
    )
    P3 = _max(P2, P3)  # for safety, use one less step as lower bound
    return P3 + 3.5 * (P3 - P2) - 0.5 * (P2 - P1)


def _critical_price_european_put(X: Float64, r: Float64, sigma: Float64, tau: Float64) -> Float64:
    """The price S that solves S = X - p(S, X, tau, r, sigma), by Newton steps.

    At this price the holder of a put that can be exercised now or at one
    later date, tau from now, is indifferent to exercising now. g(S) is
    convex and increasing, so the iteration started at X converges from
    above.
    """
    var Si = X
    var S_bar = X
    var g: Float64 = 1.0
    while abs(g) > ACCURACY:
        g = Si - X + option_price_put_black_scholes(Si, X, r, sigma, tau)
        var gprime = 1.0 + option_price_delta_put_black_scholes(Si, X, r, sigma, tau)
        S_bar = Si
        Si = Si - g / gprime
    return S_bar


def option_price_american_put_approximated_geske_johnson_corrected(
    S: Float64, X: Float64, r: Float64, sigma: Float64, time: Float64
) -> Float64:
    """Geske and Johnson (1984) approximation of an American put.

    C++ Code 15.2 with the formulas of section 15.2 applied: P1, P2 and P3
    are the prices of puts with one, two and three equally spaced exercise
    dates and the American price is extrapolated as
    P3 + 7/2 (P3 - P2) - 1/2 (P2 - P1). Changes from the listing:

    - the critical price at 2T/3 solves S = X - p(S, X, T/3), with T/3 (not
      2T/3) left to maturity;
    - the critical price at T/3 solves S = X - P2(S, X, 2T/3), the put with
      two exercise dates left (the listing uses a European put with T/3);
    - exercise at 2T/3 is discounted with exp(-r 2T/3) and exercise at T
      with exp(-r T) (interchanged in the listing);
    - the exercise price terms use d2 and the stock price terms d1 in the
      trivariate normals (interchanged in the listing).
    """
    var P1 = option_price_put_black_scholes(S, X, r, sigma, time)
    var rho12 = 1.0 / sqrt(2.0)
    var rho13 = 1.0 / sqrt(3.0)
    var rho23 = sqrt(2.0 / 3.0)
    var t2 = time / 2.0
    var t23 = time * 2.0 / 3.0
    var t3 = time / 3.0
    # two exercise dates: T/2 and T
    var S2_bar = _critical_price_european_put(X, r, sigma, t2)
    var P2 = _calc_P2(S, X, r, sigma, time, t2, S2_bar, rho12)
    P2 = _max(P1, P2)  # for safety, use one less step as lower bound
    # three exercise dates: T/3, 2T/3 and T
    var S23_bar = _critical_price_european_put(X, r, sigma, t3)  # at 2T/3, one date left
    # at T/3 two dates are left: solve S = X - P2(S, X, 2T/3) by bisection.
    # g(S) = S - X + P2(S) increases in S, is negative near zero and positive at X.
    var low = 1e-8 * X
    var high = X
    var S3_bar = 0.5 * (low + high)
    for _ in range(200):
        S3_bar = 0.5 * (low + high)
        var g = S3_bar - X + _calc_P2(S3_bar, X, r, sigma, t23, t3, S23_bar, rho12)
        if abs(g) <= ACCURACY or high - low <= 1e-12 * X:
            break
        if g < 0.0:
            low = S3_bar
        else:
            high = S3_bar
    var P3 = X * exp(-r * t3) * N(-_d2(S, S3_bar, r, sigma, t3))
    P3 -= S * N(-_d1(S, S3_bar, r, sigma, t3))
    P3 += X * exp(-r * t23) * N2(_d2(S, S3_bar, r, sigma, t3), -_d2(S, S23_bar, r, sigma, t23), -rho12)
    P3 -= S * N2(_d1(S, S3_bar, r, sigma, t3), -_d1(S, S23_bar, r, sigma, t23), -rho12)
    P3 += (
        X
        * exp(-r * time)
        * N3(
            _d2(S, S3_bar, r, sigma, t3),
            _d2(S, S23_bar, r, sigma, t23),
            -_d2(S, X, r, sigma, time),
            rho12,
            -rho13,
            -rho23,
        )
    )
    P3 -= S * N3(
        _d1(S, S3_bar, r, sigma, t3),
        _d1(S, S23_bar, r, sigma, t23),
        -_d1(S, X, r, sigma, time),
        rho12,
        -rho13,
        -rho23,
    )
    P3 = _max(P2, P3)  # for safety, use one less step as lower bound
    return P3 + 3.5 * (P3 - P2) - 0.5 * (P2 - P1)


def _baw_call(S: Float64, X: Float64, r: Float64, b: Float64, q: Float64, sigma: Float64, time: Float64) -> Float64:
    """The body of C++ Code 15.3.

    b is the cost of carry used in the quadratic approximation and q the
    payout rate handed to the European call formula. The listing passes b
    for both; the formula needs q = r - b.
    """
    var sigma_sqr = sigma * sigma
    var time_sqrt = sqrt(time)
    var nn = 2.0 * b / sigma_sqr
    var m = 2.0 * r / sigma_sqr
    var K = 1.0 - exp(-r * time)
    var q2 = (-(nn - 1.0) + sqrt((nn - 1.0) * (nn - 1.0) + (4.0 * m / K))) * 0.5
    var q2_inf = 0.5 * (-(nn - 1.0) + sqrt((nn - 1.0) * (nn - 1.0) + 4.0 * m))  # seed value from paper
    var S_star_inf = X / (1.0 - 1.0 / q2_inf)
    var h2 = -(b * time + 2.0 * sigma * time_sqrt) * (X / (S_star_inf - X))
    var S_seed = X + (S_star_inf - X) * (1.0 - exp(h2))
    var no_iterations = 0  # iterate on S to find Sstar, using Newton steps
    var Si = S_seed
    var g: Float64 = 1.0
    var gprime: Float64 = 1.0
    # the second condition is to avoid exploding Newton's
    while abs(g) > ACCURACY and abs(gprime) > ACCURACY and no_iterations < 500 and Si > 0.0:
        no_iterations += 1
        var c = _option_price_european_call_payout(Si, X, r, q, sigma, time)
        var d1 = (log(Si / X) + (b + 0.5 * sigma_sqr) * time) / (sigma * time_sqrt)
        g = (1.0 - 1.0 / q2) * Si - X - c + (1.0 / q2) * Si * exp((b - r) * time) * N(d1)
        gprime = (1.0 - 1.0 / q2) * (1.0 - exp((b - r) * time) * N(d1)) + (1.0 / q2) * exp((b - r) * time) * n(
            d1
        ) * (1.0 / (sigma * time_sqrt))
        Si = Si - (g / gprime)
    var S_star: Float64
    if abs(g) > ACCURACY:
        S_star = S_seed  # did not converge
    else:
        S_star = Si
    var C: Float64
    var c = _option_price_european_call_payout(S, X, r, q, sigma, time)
    if S >= S_star:
        C = S - X
    else:
        var d1 = (log(S_star / X) + (b + 0.5 * sigma_sqr) * time) / (sigma * time_sqrt)
        var A2 = (1.0 - exp((b - r) * time) * N(d1)) * (S_star / q2)
        C = c + A2 * pow(S / S_star, q2)
    return _max(C, c)  # know value will never be less than BS value


def option_price_american_call_approximated_baw(
    S: Float64, X: Float64, r: Float64, b: Float64, sigma: Float64, time: Float64
) -> Float64:
    """Barone-Adesi and Whaley quadratic approximation of an American call, as listed (C++ Code 15.3).

    The listing uses b both as the cost of carry and as the payout rate of
    the European call, see the module docstring. Use
    `option_price_american_call_approximated_baw_corrected` for the
    approximation of Barone-Adesi and Whaley (1987).
    """
    return _baw_call(S, X, r, b, b, sigma, time)


def option_price_american_call_approximated_baw_corrected(
    S: Float64, X: Float64, r: Float64, b: Float64, sigma: Float64, time: Float64
) -> Float64:
    """Barone-Adesi and Whaley (1987) quadratic approximation of an American call.

    b is the cost of carry of the underlying (b = r - q for a payout rate q;
    b = 0 for a futures contract). C++ Code 15.3 with the European call
    priced for the payout rate r - b. When b >= r early exercise is never
    optimal and the European price is returned.
    """
    if b >= r:
        return _option_price_european_call_payout(S, X, r, r - b, sigma, time)
    return _baw_call(S, X, r, b, r - b, sigma, time)


def _phi(
    S: Float64, T: Float64, gamma: Float64, H: Float64, X: Float64, r: Float64, b: Float64, sigma: Float64
) -> Float64:
    """The function phi of Formula 15.3 (`phi` of C++ Code 15.4)."""
    var sigma_sqr = sigma * sigma
    var kappa = 2.0 * b / sigma_sqr + 2.0 * gamma - 1.0
    var lambda_ = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma_sqr) * T
    var d1 = -(log(S / H) + (b + (gamma - 0.5) * sigma_sqr) * T) / (sigma * sqrt(T))
    var d2 = -(log((X * X) / (S * H)) + (b + (gamma - 0.5) * sigma_sqr) * T) / (sigma * sqrt(T))
    return exp(lambda_) * pow(S, gamma) * (N(d1) - pow(X / S, kappa) * N(d2))


def option_price_american_call_approximated_bjerksund_stensland(
    S: Float64, K: Float64, r: Float64, b: Float64, sigma: Float64, T: Float64
) -> Float64:
    """Bjerksund and Stensland (1993) approximation of an American call, as listed (C++ Code 15.4).

    The listing does not scale with the units of S and K and takes its
    European lower bound with b as the payout rate, see the module
    docstring. Use
    `option_price_american_call_approximated_bjerksund_stensland_corrected`
    for the approximation of Bjerksund and Stensland (1993).
    """
    var sigma_sqr = sigma * sigma
    var B0 = _max(K, r / (r - b) * K)
    var beta = (0.5 - b / sigma_sqr) + sqrt((b / sigma_sqr - 0.5) * (b / sigma_sqr - 0.5) + 2.0 * r / sigma_sqr)
    var Binf = beta / (beta - 1.0) * K
    var hT = -(b * T + 2.0 * sigma * sqrt(T)) * ((K * K) / (Binf - B0))
    var XT = B0 + (Binf - B0) * (1.0 - exp(hT))
    var alpha = (XT - K) * pow(XT, -beta)
    var C = alpha * pow(S, beta)
    C -= alpha * _phi(S, T, beta, XT, XT, r, b, sigma)
    C += _phi(S, T, 1.0, XT, XT, r, b, sigma)
    C -= _phi(S, T, 1.0, K, XT, r, b, sigma)
    C -= K * _phi(S, T, 0.0, XT, XT, r, b, sigma)
    C += K * _phi(S, T, 0.0, K, XT, r, b, sigma)
    var c = _option_price_european_call_payout(S, K, r, b, sigma, T)  # for safety use the Black Scholes as lower bound
    return _max(c, C)


def option_price_american_put_approximated_bjerksund_stensland(
    S: Float64, X: Float64, r: Float64, q: Float64, sigma: Float64, T: Float64
) -> Float64:
    """Bjerksund and Stensland (1993) approximation of an American put, as listed (C++ Code 15.5).

    The listing passes the cost of carry r - q to the call routine with the
    wrong sign and inherits the problems of C++ Code 15.4, see the module
    docstring. Use
    `option_price_american_put_approximated_bjerksund_stensland_corrected`
    for the approximation of Bjerksund and Stensland (1993).
    """
    return option_price_american_call_approximated_bjerksund_stensland(X, S, r - (r - q), r - q, sigma, T)


def option_price_american_call_approximated_bjerksund_stensland_corrected(
    S: Float64, K: Float64, r: Float64, b: Float64, sigma: Float64, T: Float64
) -> Float64:
    """Bjerksund and Stensland (1993) approximation of an American call.

    b is the cost of carry of the underlying (b = r - q for a payout rate
    q). C++ Code 15.4 with Formula 15.3 applied: h(T) uses B0 / (Binf - B0),
    the European lower bound is priced for the payout rate r - b, and the
    option is exercised at once when S is at or above the trigger price
    X_T. When b >= r early exercise is never optimal and the European price
    is returned.
    """
    var c = _option_price_european_call_payout(S, K, r, r - b, sigma, T)
    if b >= r:
        return c
    var sigma_sqr = sigma * sigma
    var B0 = _max(K, r / (r - b) * K)
    var beta = (0.5 - b / sigma_sqr) + sqrt((b / sigma_sqr - 0.5) * (b / sigma_sqr - 0.5) + 2.0 * r / sigma_sqr)
    var Binf = beta / (beta - 1.0) * K
    var hT = -(b * T + 2.0 * sigma * sqrt(T)) * (B0 / (Binf - B0))
    var XT = B0 + (Binf - B0) * (1.0 - exp(hT))
    if S >= XT:
        return S - K
    var alpha = (XT - K) * pow(XT, -beta)
    var C = alpha * pow(S, beta)
    C -= alpha * _phi(S, T, beta, XT, XT, r, b, sigma)
    C += _phi(S, T, 1.0, XT, XT, r, b, sigma)
    C -= _phi(S, T, 1.0, K, XT, r, b, sigma)
    C -= K * _phi(S, T, 0.0, XT, XT, r, b, sigma)
    C += K * _phi(S, T, 0.0, K, XT, r, b, sigma)
    return _max(c, C)  # for safety use the Black Scholes as lower bound


def option_price_american_put_approximated_bjerksund_stensland_corrected(
    S: Float64, X: Float64, r: Float64, q: Float64, sigma: Float64, T: Float64
) -> Float64:
    """Bjerksund and Stensland (1993) approximation of an American put.

    q is the payout rate of the underlying, so its cost of carry is
    b = r - q. Uses P(S, X, T, r, b, sigma) = C(X, S, T, r - b, -b, sigma).
    """
    return option_price_american_call_approximated_bjerksund_stensland_corrected(X, S, q, q - r, sigma, T)
