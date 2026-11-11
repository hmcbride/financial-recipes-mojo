"""Chapter 26: term structure derivatives (C++ Code 26.1)."""

from std.math import exp, log, sqrt
from .normdist import N
from .term_structure_models import term_structure_discount_factor_vasicek


def bond_option_price_call_zero_vasicek(
    X: Float64,
    r: Float64,
    option_time_to_maturity: Float64,
    bond_time_to_maturity: Float64,
    a: Float64,
    b: Float64,
    sigma: Float64,
) -> Float64:
    """European call on a zero coupon bond in the Vasicek model, Jamshidian (1989) (C++ Code 26.1).

    X: exercise price, r: current interest rate, a, b, sigma: parameters of
    the short rate process dr = a(b - r)dt + sigma dZ.
    """
    var T_t = option_time_to_maturity
    var s_t = bond_time_to_maturity
    var T_s = s_t - T_t
    var sigma_P: Float64
    if a == 0.0:
        # v(t,T) = sigma * sqrt(T_t); the listing computes it but does not use it
        sigma_P = sigma * T_s * sqrt(T_t)
    else:
        var v_t_T = sqrt(sigma * sigma * (1.0 - exp(-2.0 * a * T_t)) / (2.0 * a))
        var B_T_s = (1.0 - exp(-a * T_s)) / a
        sigma_P = v_t_T * B_T_s
    var h = (1.0 / sigma_P) * log(
        term_structure_discount_factor_vasicek(s_t, r, a, b, sigma)
        / (term_structure_discount_factor_vasicek(T_t, r, a, b, sigma) * X)
    ) + sigma_P / 2.0
    var c = (
        term_structure_discount_factor_vasicek(s_t, r, a, b, sigma) * N(h)
        - X * term_structure_discount_factor_vasicek(T_t, r, a, b, sigma) * N(h - sigma_P)
    )
    return c
