"""Appendix A: normal distribution approximations (C++ Code A.1 - A.3)."""

from std.math import exp, sqrt

comptime PI: Float64 = 3.141592653589793238462643


def n(z: Float64) -> Float64:
    """Normal density function (C++ Code A.1)."""
    return (1.0 / sqrt(2.0 * PI)) * exp(-0.5 * z * z)


def N(z: Float64) -> Float64:
    """Cumulative normal, Abramowitz and Stegun approximation (C++ Code A.2)."""
    if z > 6.0:
        return 1.0  # guards against overflow
    if z < -6.0:
        return 0.0
    var b1: Float64 = 0.31938153
    var b2: Float64 = -0.356563782
    var b3: Float64 = 1.781477937
    var b4: Float64 = -1.821255978
    var b5: Float64 = 1.330274429
    var p: Float64 = 0.2316419
    var c2: Float64 = 0.3989423
    var a = abs(z)
    var t = 1.0 / (1.0 + a * p)
    var b = c2 * exp((-z) * (z / 2.0))
    var nn = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t
    nn = 1.0 - b * nn
    if z < 0.0:
        nn = 1.0 - nn
    return nn


def _f(x: Float64, y: Float64, aprime: Float64, bprime: Float64, rho: Float64) -> Float64:
    var r = aprime * (2.0 * x - aprime) + bprime * (2.0 * y - bprime) + 2.0 * rho * (x - aprime) * (y - bprime)
    return exp(r)


def sgn(x: Float64) -> Float64:
    """Sign function: 1 for x >= 0, otherwise -1."""
    if x >= 0.0:
        return 1.0
    return -1.0


def N2(a: Float64, b: Float64, rho: Float64) -> Float64:
    """Cumulative bivariate normal P(x<a, y<b) with correlation rho (C++ Code A.3).

    The book overloads `N` for this; Mojo keeps it as `N2` for clarity.
    """
    if a <= 0.0 and b <= 0.0 and rho <= 0.0:
        var aprime = a / sqrt(2.0 * (1.0 - rho * rho))
        var bprime = b / sqrt(2.0 * (1.0 - rho * rho))
        var A: List[Float64] = [0.3253030, 0.4211071, 0.1334425, 0.006374323]
        var B: List[Float64] = [0.1337764, 0.6243247, 1.3425378, 2.2626645]
        var sum: Float64 = 0.0
        for i in range(4):
            for j in range(4):
                sum += A[i] * A[j] * _f(B[i], B[j], aprime, bprime, rho)
        sum = sum * (sqrt(1.0 - rho * rho) / PI)
        return sum
    elif a * b * rho <= 0.0:
        if a <= 0.0 and b >= 0.0 and rho >= 0.0:
            return N(a) - N2(a, -b, -rho)
        elif a >= 0.0 and b <= 0.0 and rho >= 0.0:
            return N(b) - N2(-a, b, -rho)
        elif a >= 0.0 and b >= 0.0 and rho <= 0.0:
            return N(a) + N(b) - 1.0 + N2(-a, -b, rho)
    elif a * b * rho >= 0.0:
        var denum = sqrt(a * a - 2.0 * rho * a * b + b * b)
        var rho1 = ((rho * a - b) * sgn(a)) / denum
        var rho2 = ((rho * b - a) * sgn(b)) / denum
        var delta = (1.0 - sgn(a) * sgn(b)) / 4.0
        return N2(a, 0.0, rho1) + N2(b, 0.0, rho2) - delta
    return -99.9  # should never get here
