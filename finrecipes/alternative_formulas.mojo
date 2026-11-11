"""Chapter 19: alternatives to the Black Scholes type option formula
(C++ Code 19.1 and 19.2).

The Heston listing leans on two C++ facilities that are replaced here:

- `std::complex<double>`: the small `Complex` struct below, with the
  arithmetic and the principal `sqrt`, `exp` and `log` the integrand needs.
- GSL's `gsl_integration_qagiu`: `integration_qagiu`, which uses the same
  change of variable to map [a, infinity) onto (0, 1] and the same 15 point
  Gauss-Kronrod rule on adaptively bisected subintervals.

`lambda` is a reserved word in Mojo, so the book's `lambda` arguments are
named `lam`.
"""

from std.math import exp, log, sqrt, pow, cos, sin, atan2, hypot, isnan, isinf
from .normdist import PI
from .black_scholes import option_price_call_black_scholes


def option_price_call_merton_jump_diffusion(
    S: Float64,
    X: Float64,
    r: Float64,
    sigma: Float64,
    time_to_maturity: Float64,
    lam: Float64,
    kappa: Float64,
    delta: Float64,
) -> Float64:
    """European call in Merton's jump diffusion model (C++ Code 19.1).

    S: price of the underlying, X: exercise price, r: interest rate,
    sigma: volatility of the diffusion part, time_to_maturity: time to
    maturity, lam: jump intensity (the book's lambda), kappa: expected
    relative jump size, delta: standard deviation of the log jump size.
    The infinite sum is cut off after MAXN = 50 terms.
    """
    var MAXN = 50
    var tau = time_to_maturity
    var sigma_sqr = sigma * sigma
    var delta_sqr = delta * delta
    var lambdaprime = lam * (1.0 + kappa)
    var gamma = log(1.0 + kappa)
    var c = exp(-lambdaprime * tau) * option_price_call_black_scholes(S, X, r - lam * kappa, sigma, tau)
    var log_n: Float64 = 0.0
    for n in range(1, MAXN + 1):
        log_n += log(Float64(n))
        var sigma_n = sqrt(sigma_sqr + Float64(n) * delta_sqr / tau)
        var r_n = r - lam * kappa + Float64(n) * gamma / tau
        c += exp(
            -lambdaprime * tau + Float64(n) * log(lambdaprime * tau) - log_n
        ) * option_price_call_black_scholes(S, X, r_n, sigma_n, tau)
    return c


struct Complex(Copyable, Movable):
    """A complex number, standing in for C++ `complex<double>`."""

    var re: Float64
    var im: Float64

    def __init__(out self, re: Float64, im: Float64):
        self.re = re
        self.im = im

    def __add__(self, other: Complex) -> Complex:
        return Complex(self.re + other.re, self.im + other.im)

    def __sub__(self, other: Complex) -> Complex:
        return Complex(self.re - other.re, self.im - other.im)

    def __mul__(self, other: Complex) -> Complex:
        return Complex(self.re * other.re - self.im * other.im, self.re * other.im + self.im * other.re)

    def __mul__(self, other: Float64) -> Complex:
        return Complex(self.re * other, self.im * other)

    def __truediv__(self, other: Complex) -> Complex:
        var denom = other.re * other.re + other.im * other.im
        return Complex(
            (self.re * other.re + self.im * other.im) / denom, (self.im * other.re - self.re * other.im) / denom
        )


def complex_sqrt(z: Complex) -> Complex:
    """Principal square root (non-negative real part)."""
    var modulus = hypot(z.re, z.im)
    if modulus == 0.0:
        return Complex(0.0, 0.0)
    var t = sqrt(0.5 * (abs(z.re) + modulus))
    if z.re >= 0.0:
        return Complex(t, z.im / (2.0 * t))
    if z.im >= 0.0:
        return Complex(abs(z.im) / (2.0 * t), t)
    return Complex(abs(z.im) / (2.0 * t), -t)


def complex_exp(z: Complex) -> Complex:
    var e = exp(z.re)
    return Complex(e * cos(z.im), e * sin(z.im))


def complex_log(z: Complex) -> Complex:
    """Principal logarithm (imaginary part in (-pi, pi])."""
    return Complex(log(hypot(z.re, z.im)), atan2(z.im, z.re))


trait RealFunction:
    """A real function of one variable, the role of `gsl_function` in the book."""

    def eval(self, x: Float64) -> Float64:
        ...


def _rescale_error(err_in: Float64, result_abs: Float64, result_asc: Float64) -> Float64:
    """The error heuristic of QUADPACK's Gauss-Kronrod rules."""
    var DBL_EPSILON: Float64 = 2.2204460492503131e-16
    var DBL_MIN: Float64 = 2.2250738585072014e-308
    var err = abs(err_in)
    if result_asc != 0.0 and err != 0.0:
        var scale = pow(200.0 * err / result_asc, 1.5)
        if scale < 1.0:
            err = result_asc * scale
        else:
            err = result_asc
    if result_abs > DBL_MIN / (50.0 * DBL_EPSILON):
        var min_err = 50.0 * DBL_EPSILON * result_abs
        if min_err > err:
            err = min_err
    return err


def _qk15_upper[F: RealFunction](f: F, lower: Float64, a: Float64, b: Float64) -> Tuple[Float64, Float64]:
    """15 point Gauss-Kronrod rule on the transformed integrand.

    Integrates g(t) = f(lower + (1 - t) / t) / t^2 over [a, b], a subinterval
    of (0, 1], and returns (integral, error estimate). The error estimate
    compares the Kronrod result with the embedded 7 point Gauss result.
    """
    # abscissae of the 15 point Kronrod rule; xgk[1], xgk[3], xgk[5], xgk[7]
    # are the abscissae of the 7 point Gauss rule
    var xgk: List[Float64] = [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144838258730,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.000000000000000000000000000000000,
    ]
    var wgk: List[Float64] = [  # weights of the 15 point Kronrod rule
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
    ]
    var wg: List[Float64] = [  # weights of the 7 point Gauss rule
        0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,
        0.381830050505118944950369775488975,
        0.417959183673469387755102040816327,
    ]
    var center = 0.5 * (a + b)
    var half_length = 0.5 * (b - a)
    var f_center = f.eval(lower + (1.0 - center) / center) / (center * center)
    var fv1 = List[Float64](length=7, fill=0.0)
    var fv2 = List[Float64](length=7, fill=0.0)
    var result_gauss = f_center * wg[3]
    var result_kronrod = f_center * wgk[7]
    var result_abs = abs(result_kronrod)
    for j in range(7):
        var abscissa = half_length * xgk[j]
        var t1 = center - abscissa
        var t2 = center + abscissa
        var fval1 = f.eval(lower + (1.0 - t1) / t1) / (t1 * t1)
        var fval2 = f.eval(lower + (1.0 - t2) / t2) / (t2 * t2)
        fv1[j] = fval1
        fv2[j] = fval2
        if j % 2 == 1:
            result_gauss += wg[j // 2] * (fval1 + fval2)
        result_kronrod += wgk[j] * (fval1 + fval2)
        result_abs += wgk[j] * (abs(fval1) + abs(fval2))
    var mean = result_kronrod * 0.5
    var result_asc = wgk[7] * abs(f_center - mean)
    for j in range(7):
        result_asc += wgk[j] * (abs(fv1[j] - mean) + abs(fv2[j] - mean))
    var err = (result_kronrod - result_gauss) * half_length
    result_kronrod *= half_length
    result_abs *= abs(half_length)
    result_asc *= abs(half_length)
    return (result_kronrod, _rescale_error(err, result_abs, result_asc))


def integration_qagiu[F: RealFunction](
    f: F, a: Float64, epsabs: Float64, epsrel: Float64, limit: Int
) -> Float64:
    """Adaptive integration of f over [a, infinity).

    Replaces `gsl_integration_qagiu` in C++ Code 19.2. As in QUADPACK, the
    substitution x = a + (1 - t) / t turns the integral into one over
    (0, 1], which is estimated with the 15 point Gauss-Kronrod rule. The
    subinterval with the largest error estimate is bisected until the total
    error estimate is below max(epsabs, epsrel |integral|), or `limit`
    subintervals are in use, or the worst subinterval has become too small
    to bisect. The best estimate is returned in all three cases; like the
    book's code, which ignores the GSL status, no error is reported.

    GSL also extrapolates the sequence of estimates to speed up convergence
    for integrands with end point singularities. That step is left out, so
    this routine is meant for integrands that are smooth on the open
    interval, such as Heston's.
    """
    var lows = List[Float64]()
    var highs = List[Float64]()
    var results = List[Float64]()
    var errors = List[Float64]()
    var first = _qk15_upper(f, a, 0.0, 1.0)
    lows.append(0.0)
    highs.append(1.0)
    results.append(first[0])
    errors.append(first[1])
    var result = first[0]
    var errsum = first[1]
    while errsum > max(epsabs, epsrel * abs(result)) and len(lows) < limit:
        var worst = 0  # the subinterval with the largest error estimate
        for k in range(1, len(errors)):
            if errors[k] > errors[worst]:
                worst = k
        var low = lows[worst]
        var high = highs[worst]
        var mid = 0.5 * (low + high)
        # QUADPACK's test for a subinterval too small to bisect: both end
        # points agree with the midpoint to about 100 rounding units
        var tmp = (1.0 + 100.0 * 2.2204460492503131e-16) * (abs(mid) + 1000.0 * 2.2250738585072014e-308)
        if (abs(low) <= tmp and abs(high) <= tmp) or mid <= low or mid >= high:
            break  # the estimate cannot be improved any further
        var left = _qk15_upper(f, a, low, mid)
        var right = _qk15_upper(f, a, mid, high)
        highs[worst] = mid
        results[worst] = left[0]
        errors[worst] = left[1]
        lows.append(mid)
        highs.append(high)
        results.append(right[0])
        errors.append(right[1])
        result = 0.0
        errsum = 0.0
        for k in range(len(results)):
            result += results[k]
            errsum += errors[k]
    return result


struct HestonParms(RealFunction, Copyable, Movable):
    """Parameters of the Heston integrand (`struct heston_parms` in C++ Code 19.2).

    `eval` makes the struct usable as the integrand of `integration_qagiu`.
    """

    var K: Float64
    var x: Float64
    var r: Float64
    var v: Float64
    var tau: Float64
    var kappa: Float64
    var theta: Float64
    var rho: Float64
    var sigma: Float64
    var lam: Float64
    var j: Int

    def __init__(
        out self,
        K: Float64,
        x: Float64,
        r: Float64,
        v: Float64,
        tau: Float64,
        kappa: Float64,
        theta: Float64,
        rho: Float64,
        sigma: Float64,
        lam: Float64,
        j: Int,
    ):
        self.K = K
        self.x = x
        self.r = r
        self.v = v
        self.tau = tau
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.sigma = sigma
        self.lam = lam
        self.j = j

    def eval(self, x: Float64) -> Float64:
        return heston_integrand_j(x, self)


def heston_integrand_j(phi: Float64, parms: HestonParms) -> Float64:
    """Integrand Re(e^(-i phi ln K) f_j / (i phi)) of Heston's P_j (C++ Code 19.2).

    For very large phi the term exp(d tau) overflows and the expression
    evaluates to inf/inf. The integrand itself tends to zero there, so a
    result that is not finite is returned as 0.
    """
    var K = parms.K
    var x = parms.x
    var v = parms.v
    var r = parms.r
    var kappa = parms.kappa
    var theta = parms.theta
    var rho = parms.rho
    var sigma = parms.sigma
    var lam = parms.lam
    var tau = parms.tau
    var sigma_sqr = sigma * sigma
    var uj: Float64 = -0.5
    var bj = kappa + lam
    if parms.j == 1:
        uj = 0.5
        bj = kappa + lam - rho * sigma
    var one = Complex(1.0, 0.0)
    var a = kappa * theta
    # rho*sigma*phi*i - bj
    var rspi_minus_bj = Complex(-bj, rho * sigma * phi)
    # bj - rho*sigma*phi*i
    var bj_minus_rspi = Complex(bj, -rho * sigma * phi)
    # d = sqrt( pow(rho*sigma*phi*i-bj,2) - sigma_sqr*(2*uj*phi*i-pow(phi,2)) )
    var d = complex_sqrt(
        rspi_minus_bj * rspi_minus_bj - Complex(-sigma_sqr * phi * phi, sigma_sqr * 2.0 * uj * phi)
    )
    # g = (bj-rho*sigma*phi*i+d)/(bj-rho*sigma*phi*i-d)
    var g = (bj_minus_rspi + d) / (bj_minus_rspi - d)
    var exp_d_tau = complex_exp(d * tau)
    # C = r*phi*i*tau+(a/sigma_sqr)*((bj-rho*sigma*phi*i+d)*tau-2.0*log((1.0-g*exp(d*tau))/(1.0-g)))
    var C = Complex(0.0, r * phi * tau) + (
        (bj_minus_rspi + d) * tau - complex_log((one - g * exp_d_tau) / (one - g)) * 2.0
    ) * (a / sigma_sqr)
    # D = (bj-rho*sigma*phi*i+d)/sigma_sqr * ( (1.0-exp(d*tau))/(1.0-g*exp(d*tau)) )
    var D = ((bj_minus_rspi + d) * (1.0 / sigma_sqr)) * ((one - exp_d_tau) / (one - g * exp_d_tau))
    # f1 = exp(C+D*v+i*phi*x)
    var f1 = complex_exp(C + D * v + Complex(0.0, phi * x))
    # F = exp(-phi*i*log(K))*f1/(i*phi)
    var F = complex_exp(Complex(0.0, -phi * log(K))) * f1 / Complex(0.0, phi)
    if isnan(F.re) or isinf(F.re):
        return 0.0
    return F.re


def heston_Pj(
    S: Float64,
    K: Float64,
    r: Float64,
    v: Float64,
    tau: Float64,
    sigma: Float64,
    kappa: Float64,
    lam: Float64,
    rho: Float64,
    theta: Float64,
    j: Int,
) -> Float64:
    """The probability P_j of Heston's formula, j = 1 or 2 (C++ Code 19.2)."""
    var x = log(S)
    var parms = HestonParms(K, x, r, v, tau, kappa, theta, rho, sigma, lam, j)
    var n = 10000
    # integral to infinity starting at zero
    var result = integration_qagiu(parms, 0.0, 1e-7, 1e-7, n)
    return 0.5 + result / PI


def heston_call_option_price(
    S: Float64,
    K: Float64,
    r: Float64,
    v: Float64,
    tau: Float64,
    rho: Float64,
    kappa: Float64,
    lam: Float64,
    theta: Float64,
    sigma: Float64,
) -> Float64:
    """European call in Heston's stochastic volatility model (C++ Code 19.2).

    S: price of the underlying, K: exercise price, r: interest rate,
    v: current variance, tau: time to maturity, rho: correlation between the
    price and variance processes, kappa: speed of mean reversion of the
    variance, lam: price of volatility risk (the book's lambda), theta: long
    run variance, sigma: volatility of the variance.
    """
    var P1 = heston_Pj(S, K, r, v, tau, sigma, kappa, lam, rho, theta, 1)
    var P2 = heston_Pj(S, K, r, v, tau, sigma, kappa, lam, rho, theta, 2)
    var C = S * P1 - K * exp(-r * tau) * P2
    return C
