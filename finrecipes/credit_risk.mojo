"""Chapter 21: credit risk, the Merton model.

The chapter has no numbered listing. Its example program values the debt of
a firm as risk free debt minus a put option on the firm value, priced with
the Black Scholes formula. `debt_value_merton` is that calculation.
"""

from std.math import exp
from .black_scholes import option_price_put_black_scholes


def debt_value_merton(V: Float64, F: Float64, r: Float64, sigma: Float64, T: Float64) -> Float64:
    """Value of risky zero coupon debt in the Merton model (example program, section 21.1).

    V: current value of the firm, F: face value of the debt, r: risk free
    interest rate, sigma: volatility of the firm value, T: time to maturity
    of the debt.
    """
    var p = option_price_put_black_scholes(V, F, r, sigma, T)
    return exp(-r * T) * F - p
