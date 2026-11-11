"""Chapter 5: the term structure of interest rates (C++ Code 5.1 - 5.8).

The C++ base class `term_structure_class` becomes the `TermStructure` trait.
A concrete term structure must override at least one of `r` or `d`; the
default methods are defined in terms of each other, exactly as in the book.
"""

from std.math import exp, log


def term_structure_yield_from_discount_factor(d_t: Float64, t: Float64) -> Float64:
    return -log(d_t) / t


def term_structure_discount_factor_from_yield(r: Float64, t: Float64) -> Float64:
    return exp(-r * t)


def term_structure_forward_rate_from_discount_factors(d_t1: Float64, d_t2: Float64, time: Float64) -> Float64:
    return log(d_t1 / d_t2) / time


def term_structure_forward_rate_from_yields(r_t1: Float64, r_t2: Float64, t1: Float64, t2: Float64) -> Float64:
    return r_t2 * t2 / (t2 - t1) - r_t1 * t1 / (t2 - t1)


trait TermStructure:
    """Generic term structure: spot rates r, discount factors d, forward rates f."""

    def r(self, t: Float64) -> Float64:
        """Yield on a zero coupon bond maturing at t."""
        return term_structure_yield_from_discount_factor(self.d(t), t)

    def d(self, t: Float64) -> Float64:
        """Discount factor (price of a zero coupon bond) for time t."""
        return term_structure_discount_factor_from_yield(self.r(t), t)

    def f(self, t1: Float64, t2: Float64) -> Float64:
        """Forward rate between t1 and t2."""
        return term_structure_forward_rate_from_discount_factors(self.d(t1), self.d(t2), t2 - t1)


struct TermStructureFlat(TermStructure, Copyable, Movable):
    """Flat term structure (C++ Code 5.3)."""

    var R: Float64

    def __init__(out self, r: Float64):
        self.R = r

    def r(self, t: Float64) -> Float64:
        if t >= 0.0:
            return self.R
        return 0.0

    def set_int_rate(mut self, r: Float64):
        self.R = r


def term_structure_yield_linearly_interpolated(
    time: Float64, obs_times: List[Float64], obs_yields: List[Float64]
) -> Float64:
    """Linear interpolation of spot rates (C++ Code 5.4).

    Assumes the observations are sorted by increasing time to maturity.
    """
    var no_obs = len(obs_times)
    if no_obs < 1:
        return 0.0
    var t_min = obs_times[0]
    if time <= t_min:
        return obs_yields[0]  # earlier than lowest obs.
    var t_max = obs_times[no_obs - 1]
    if time >= t_max:
        return obs_yields[no_obs - 1]  # later than latest obs
    var t = 1  # find which two observations we are between
    while t < no_obs and time > obs_times[t]:
        t += 1
    var lam = (obs_times[t] - time) / (obs_times[t] - obs_times[t - 1])
    return obs_yields[t - 1] * lam + obs_yields[t] * (1.0 - lam)


struct TermStructureInterpolated(TermStructure, Copyable, Movable):
    """Term structure interpolating linearly between observed spot rates (C++ Code 5.5)."""

    var times: List[Float64]
    var yields: List[Float64]

    def __init__(out self):
        self.times = List[Float64]()
        self.yields = List[Float64]()

    def __init__(out self, times: List[Float64], yields: List[Float64]):
        self.times = List[Float64]()
        self.yields = List[Float64]()
        if len(times) == len(yields):
            self.times = times.copy()
            self.yields = yields.copy()

    def no_observations(self) -> Int:
        return len(self.times)

    def r(self, t: Float64) -> Float64:
        return term_structure_yield_linearly_interpolated(t, self.times, self.yields)

    def set_interpolated_observations(mut self, times: List[Float64], yields: List[Float64]):
        self.times = List[Float64]()
        self.yields = List[Float64]()
        if len(times) == len(yields):
            self.times = times.copy()
            self.yields = yields.copy()


def bonds_price[T: TermStructure](cashflow_times: List[Float64], cashflows: List[Float64], d: T) -> Float64:
    """Bond price with a general term structure (C++ Code 5.6)."""
    var p: Float64 = 0.0
    for i in range(len(cashflow_times)):
        p += d.d(cashflow_times[i]) * cashflows[i]
    return p


def bonds_duration[T: TermStructure](
    cashflow_times: List[Float64], cashflow_amounts: List[Float64], d: T
) -> Float64:
    """Bond duration with a general term structure (C++ Code 5.7)."""
    var S: Float64 = 0.0
    var D1: Float64 = 0.0
    for i in range(len(cashflow_times)):
        S += cashflow_amounts[i] * d.d(cashflow_times[i])
        D1 += cashflow_times[i] * cashflow_amounts[i] * d.d(cashflow_times[i])
    return D1 / S


def bonds_convexity[T: TermStructure](
    cashflow_times: List[Float64], cashflow_amounts: List[Float64], d: T
) -> Float64:
    """Bond convexity with a general term structure (C++ Code 5.8)."""
    var B: Float64 = 0.0
    var Cx: Float64 = 0.0
    for i in range(len(cashflow_times)):
        B += cashflow_amounts[i] * d.d(cashflow_times[i])
        Cx += cashflow_times[i] * cashflow_times[i] * cashflow_amounts[i] * d.d(cashflow_times[i])
    return Cx / B
