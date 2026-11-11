"""Chapter 7: futures algorithms (C++ Code 7.1)."""

from std.math import exp


def futures_price(S: Float64, r: Float64, time_to_maturity: Float64) -> Float64:
    """Futures price of an asset without payouts (C++ Code 7.1).

    S: current price of underlying asset, r: risk free interest rate,
    time_to_maturity: time to maturity of the contract.
    """
    return exp(r * time_to_maturity) * S
