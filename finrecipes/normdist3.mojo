"""Appendix C: the trivariate cumulative normal N3 (C++ Code C.1).

The book evaluates the univariate integral of Geske and Johnson (1984)

    N3(h, k, j; rho12, rho13, rho23)
        = integral from -infinity to j of
          n(z) * N2((k - rho23 z) / sqrt(1 - rho23^2),
                    (h - rho13 z) / sqrt(1 - rho13^2),
                    (rho12 - rho13 rho23) / (sqrt(1 - rho13^2) sqrt(1 - rho23^2))) dz

with the GNU Scientific Library routine `gsl_integration_qags` on [-20, j],
absolute and relative tolerance 1e-7 and at most 1000 subintervals. GSL is
not available from Mojo, so the quadrature is written out here: the same
21 point Gauss-Kronrod rule that `qags` applies to each subinterval, the
QUADPACK error estimate, and bisection of the subinterval with the largest
error estimate until the tolerance is met. (`qags` additionally accelerates
convergence with the epsilon algorithm; that only matters for singular
integrands, which this one is not.)

The accuracy of N3 is limited by the bivariate normal approximation `N2`
of Appendix A (about four decimals), not by the quadrature.
"""

from std.math import sqrt
from .normdist import n, N2

comptime LOWER_LIMIT: Float64 = -20.0  # stands in for minus infinity
comptime EPSABS: Float64 = 1e-7
comptime EPSREL: Float64 = 1e-7
comptime MAX_INTERVALS = 1000
comptime DBL_EPSILON: Float64 = 2.220446049250313e-16
comptime DBL_MIN: Float64 = 2.2250738585072014e-308


struct N3Parms(Copyable, Movable):
    """Parameters of the integrand (the `n3_parms` struct of C++ Code C.1)."""

    var h: Float64
    var k: Float64
    var rho12: Float64
    var rho13: Float64
    var rho23: Float64

    def __init__(out self, h: Float64, k: Float64, rho12: Float64, rho13: Float64, rho23: Float64):
        self.h = h
        self.k = k
        self.rho12 = rho12
        self.rho13 = rho13
        self.rho23 = rho23


def f3(z: Float64, parms: N3Parms) -> Float64:
    """The integrand of N3 (function `f3` of C++ Code C.1)."""
    var h = parms.h
    var k = parms.k
    var rho12 = parms.rho12
    var rho13 = parms.rho13
    var rho23 = parms.rho23
    var f = n(z)
    f *= N2(
        (k - rho23 * z) / sqrt(1.0 - rho23 * rho23),
        (h - rho13 * z) / (sqrt(1.0 - rho13 * rho13)),
        (rho12 - rho13 * rho23) / (sqrt(1.0 - rho13 * rho13) * sqrt(1.0 - rho23 * rho23)),
    )
    return f


def _qk21(a: Float64, b: Float64, parms: N3Parms) -> Tuple[Float64, Float64]:
    """21 point Gauss-Kronrod rule on [a, b]: (integral, error estimate).

    Nodes, weights and the error heuristic are those of QUADPACK's QK21,
    which GSL uses inside `gsl_integration_qags`.
    """
    # abscissae of the 21 point Kronrod rule; entries 1, 3, 5, 7, 9 are the
    # abscissae of the 10 point Gauss rule
    var xgk: List[Float64] = [
        0.995657163025808080735527280689003,
        0.973906528517171720077964012084452,
        0.930157491355708226001207180059508,
        0.865063366688984510732096688423493,
        0.780817726586416897063717578345042,
        0.679409568299024406234327365114874,
        0.562757134668604683339000099272694,
        0.433395394129247190799265943165784,
        0.294392862701460198131126603103866,
        0.148874338981631210884826001129720,
        0.000000000000000000000000000000000,
    ]
    # weights of the 10 point Gauss rule
    var wg: List[Float64] = [
        0.066671344308688137593568809893332,
        0.149451349150580593145776339657697,
        0.219086362515982043995534934228163,
        0.269266719309996355091226921569469,
        0.295524224714752870173815619188769,
    ]
    # weights of the 21 point Kronrod rule
    var wgk: List[Float64] = [
        0.011694638867371874278064396062192,
        0.032558162307964727478818972459390,
        0.054755896574351996031381300244580,
        0.075039674810919952767043140916190,
        0.093125454583697605535065465083366,
        0.109387158802297641899210590325805,
        0.123491976262065851077958109585166,
        0.134709217311473325928054001771707,
        0.142775938577060080797094273138717,
        0.147739104901338491374841515972068,
        0.149445554002916905664936468389821,
    ]
    var center = 0.5 * (a + b)
    var half_length = 0.5 * (b - a)
    var abs_half_length = abs(half_length)
    var f_center = f3(center, parms)
    var fv1 = List[Float64](length=10, fill=0.0)  # values right of the center
    var fv2 = List[Float64](length=10, fill=0.0)  # values left of the center
    var result_gauss: Float64 = 0.0
    var result_kronrod = f_center * wgk[10]
    var result_abs = abs(result_kronrod)
    for i in range(10):
        var abscissa = half_length * xgk[i]
        var fval1 = f3(center - abscissa, parms)
        var fval2 = f3(center + abscissa, parms)
        fv1[i] = fval1
        fv2[i] = fval2
        result_kronrod += wgk[i] * (fval1 + fval2)
        result_abs += wgk[i] * (abs(fval1) + abs(fval2))
        if i % 2 == 1:
            result_gauss += wg[i // 2] * (fval1 + fval2)
    var mean = result_kronrod * 0.5
    var result_asc = wgk[10] * abs(f_center - mean)
    for i in range(10):
        result_asc += wgk[i] * (abs(fv1[i] - mean) + abs(fv2[i] - mean))
    var err = abs((result_kronrod - result_gauss) * half_length)
    result_kronrod *= half_length
    result_abs *= abs_half_length
    result_asc *= abs_half_length
    if result_asc != 0.0 and err != 0.0:
        var scale = (200.0 * err / result_asc) ** 1.5
        if scale < 1.0:
            err = result_asc * scale
        else:
            err = result_asc
    if result_abs > DBL_MIN / (50.0 * DBL_EPSILON):
        var min_err = 50.0 * DBL_EPSILON * result_abs
        if min_err > err:
            err = min_err
    return (result_kronrod, err)


def N3(h: Float64, k: Float64, j: Float64, rho12: Float64, rho13: Float64, rho23: Float64) -> Float64:
    """Trivariate cumulative normal P(x1<h, x2<k, x3<j) (C++ Code C.1).

    rho12, rho13, rho23 are the correlations between the three standard
    normal variables. Uses the method of Geske and Johnson (1984): a
    univariate integral over the third variable of the density times a
    bivariate cumulative normal, evaluated by adaptive quadrature.
    """
    var parms = N3Parms(h, k, rho12, rho13, rho23)
    # adaptive integration over [-20, j], the workspace of subintervals
    var lower = List[Float64]()
    var upper = List[Float64]()
    var integral = List[Float64]()
    var error = List[Float64]()
    var first = _qk21(LOWER_LIMIT, j, parms)
    lower.append(LOWER_LIMIT)
    upper.append(j)
    integral.append(first[0])
    error.append(first[1])
    var result = first[0]
    var errsum = first[1]
    while len(lower) < MAX_INTERVALS:
        var tolerance = EPSABS
        if EPSREL * abs(result) > tolerance:
            tolerance = EPSREL * abs(result)
        if errsum <= tolerance:
            break
        # bisect the subinterval with the largest error estimate
        var worst = 0
        for i in range(1, len(error)):
            if error[i] > error[worst]:
                worst = i
        var a = lower[worst]
        var b = upper[worst]
        var mid = 0.5 * (a + b)
        var left = _qk21(a, mid, parms)
        var right = _qk21(mid, b, parms)
        result += left[0] + right[0] - integral[worst]
        errsum += left[1] + right[1] - error[worst]
        upper[worst] = mid
        integral[worst] = left[0]
        error[worst] = left[1]
        lower.append(mid)
        upper.append(b)
        integral.append(right[0])
        error.append(right[1])
    return result
