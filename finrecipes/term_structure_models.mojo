"""Chapter 22: term structure models (C++ Code 22.1 - 22.10, Header files 22.1 - 22.5).

Each model is a plain function plus a wrapper struct implementing the
`TermStructure` trait of chapter 5. The Nelson Siegel and Svensson wrappers
supply the spot rate `r`; the cubic spline, Cox Ingersoll Ross and Vasicek
wrappers supply the discount factor `d`. The trait derives the rest.

The wrapper fields keep the C++ private member names with their trailing
underscore (`beta0_`, `r_`), which also keeps them apart from the trait
methods `r` and `d`. `lambda` is `lambda_` throughout.
"""

from std.math import exp, sqrt, pow
from .term_structure import TermStructure


# Nelson Siegel


def term_structure_yield_nelson_siegel(
    t: Float64, beta0: Float64, beta1: Float64, beta2: Float64, lambda_: Float64
) -> Float64:
    """Spot rate in the Nelson and Siegel (1987) term structure model (C++ Code 22.1)."""
    if t == 0.0:
        return beta0
    var tl = t / lambda_
    var r = beta0 + (beta1 + beta2) * ((1.0 - exp(-tl)) / tl) + beta2 * exp(-tl)
    return r


struct TermStructureNelsonSiegel(TermStructure, Copyable, Movable):
    """Term structure wrapper for the Nelson Siegel approximation (Header file 22.1, C++ Code 22.2)."""

    var beta0_: Float64
    var beta1_: Float64
    var beta2_: Float64
    var lambda_: Float64

    def __init__(out self, b0: Float64, b1: Float64, b2: Float64, l: Float64):
        self.beta0_ = b0
        self.beta1_ = b1
        self.beta2_ = b2
        self.lambda_ = l

    def r(self, t: Float64) -> Float64:
        if t <= 0.0:
            return self.beta0_
        return term_structure_yield_nelson_siegel(t, self.beta0_, self.beta1_, self.beta2_, self.lambda_)


# Svensson


def term_structure_yield_svensson(
    t: Float64, beta0: Float64, beta1: Float64, beta2: Float64, beta3: Float64, tau1: Float64, tau2: Float64
) -> Float64:
    """Spot rate in Svensson's extended Nelson and Siegel model (C++ Code 22.3)."""
    if t == 0.0:
        return beta0
    var r = beta0
    r += beta1 * ((1.0 - exp(-t / tau1)) / (t / tau1))
    r += beta2 * (((1.0 - exp(-t / tau1)) / (t / tau1)) - exp(-t / tau1))
    r += beta3 * (((1.0 - exp(-t / tau2)) / (t / tau2)) - exp(-t / tau2))
    return r


struct TermStructureSvensson(TermStructure, Copyable, Movable):
    """Term structure wrapper for the Svensson model (Header file 22.2, C++ Code 22.4)."""

    var beta0_: Float64
    var beta1_: Float64
    var beta2_: Float64
    var beta3_: Float64
    var tau1_: Float64
    var tau2_: Float64

    def __init__(out self, b0: Float64, b1: Float64, b2: Float64, b3: Float64, tau1: Float64, tau2: Float64):
        self.beta0_ = b0
        self.beta1_ = b1
        self.beta2_ = b2
        self.beta3_ = b3
        self.tau1_ = tau1
        self.tau2_ = tau2

    def r(self, t: Float64) -> Float64:
        if t <= 0.0:
            return self.beta0_
        return term_structure_yield_svensson(
            t, self.beta0_, self.beta1_, self.beta2_, self.beta3_, self.tau1_, self.tau2_
        )


# Cubic spline


def term_structure_discount_factor_cubic_spline(
    t: Float64, b1: Float64, c1: Float64, d1: Float64, f: List[Float64], knots: List[Float64]
) -> Float64:
    """Discount factor approximated with a cubic spline (C++ Code 22.5).

    Assumes the knots are sorted in increasing order.
    """
    var d = 1.0 + b1 * t + c1 * (t * t) + d1 * (t * t * t)
    for i in range(len(knots)):
        if t >= knots[i]:
            var tk = t - knots[i]
            d += f[i] * (tk * tk * tk)
        else:
            break
    return d


struct TermStructureCubicSpline(TermStructure, Copyable, Movable):
    """Term structure wrapper for the cubic spline approximation (Header file 22.3, C++ Code 22.6)."""

    var b_: Float64
    var c_: Float64
    var d_: Float64
    var f_: List[Float64]
    var knots_: List[Float64]

    def __init__(out self, b: Float64, c: Float64, d: Float64, f: List[Float64], knots: List[Float64]):
        self.b_ = b
        self.c_ = c
        self.d_ = d
        self.f_ = List[Float64]()
        self.knots_ = List[Float64]()
        if len(f) != len(knots):
            return
        self.f_ = f.copy()
        self.knots_ = knots.copy()

    def d(self, T: Float64) -> Float64:
        return term_structure_discount_factor_cubic_spline(T, self.b_, self.c_, self.d_, self.f_, self.knots_)


# Cox Ingersoll Ross


def term_structure_discount_factor_cir(
    t: Float64, r: Float64, kappa: Float64, lambda_: Float64, theta: Float64, sigma: Float64
) -> Float64:
    """Discount factor in the Cox, Ingersoll and Ross (1985) model (C++ Code 22.7).

    r: short interest rate, kappa: mean reversion parameter, lambda_: "market"
    risk parameter, theta: long run mean, sigma: volatility.
    """
    var sigma_sqr = sigma * sigma
    var gamma = sqrt((kappa + lambda_) * (kappa + lambda_) + 2.0 * sigma_sqr)
    var denum = (gamma + kappa + lambda_) * (exp(gamma * t) - 1.0) + 2.0 * gamma
    var p = 2.0 * kappa * theta / sigma_sqr
    var enum1 = 2.0 * gamma * exp(0.5 * (kappa + lambda_ + gamma) * t)
    var A = pow(enum1 / denum, p)
    var B = (2.0 * (exp(gamma * t) - 1.0)) / denum
    var dfact = A * exp(-B * r)
    return dfact


struct TermStructureCir(TermStructure, Copyable, Movable):
    """Term structure wrapper for the Cox, Ingersoll and Ross model (Header file 22.4, C++ Code 22.8)."""

    var r_: Float64  # interest rate
    var kappa_: Float64  # mean reversion parameter
    var lambda_: Float64  # risk aversion
    var theta_: Float64  # long run mean
    var sigma_: Float64  # volatility

    def __init__(out self, r: Float64, k: Float64, l: Float64, th: Float64, sigma: Float64):
        self.r_ = r
        self.kappa_ = k
        self.lambda_ = l
        self.theta_ = th
        self.sigma_ = sigma

    def d(self, T: Float64) -> Float64:
        return term_structure_discount_factor_cir(T, self.r_, self.kappa_, self.lambda_, self.theta_, self.sigma_)


# Vasicek


def term_structure_discount_factor_vasicek(time: Float64, r: Float64, a: Float64, b: Float64, sigma: Float64) -> Float64:
    """Discount factor with the Vasicek functional form (C++ Code 22.9).

    The short rate follows dr = a(b - r)dt + sigma dZ.

    Correction to the book: for a == 0 the listing has
    `A = exp(sigma_sqr*pow(time,3))/6.0`, with the division by 6 outside the
    exponential. That gives a discount factor of 1/6 at time 0. The limit of
    the a != 0 branch as a goes to 0 is exp(sigma^2 time^3 / 6), which is
    what is used here.
    """
    var A: Float64
    var B: Float64
    var sigma_sqr = sigma * sigma
    var aa = a * a
    if a == 0.0:
        B = time
        A = exp(sigma_sqr * (time * time * time) / 6.0)
    else:
        B = (1.0 - exp(-a * time)) / a
        A = exp(((B - time) * (aa * b - 0.5 * sigma_sqr)) / aa - ((sigma_sqr * B * B) / (4.0 * a)))
    var dfact = A * exp(-B * r)
    return dfact


struct TermStructureVasicek(TermStructure, Copyable, Movable):
    """Term structure wrapper for the Vasicek (1977) model (Header file 22.5, C++ Code 22.10)."""

    var r_: Float64
    var a_: Float64
    var b_: Float64
    var sigma_: Float64

    def __init__(out self, r: Float64, a: Float64, b: Float64, sigma: Float64):
        self.r_ = r
        self.a_ = a
        self.b_ = b
        self.sigma_ = sigma

    def d(self, T: Float64) -> Float64:
        return term_structure_discount_factor_vasicek(T, self.r_, self.a_, self.b_, self.sigma_)
