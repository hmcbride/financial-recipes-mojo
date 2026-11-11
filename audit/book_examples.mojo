"""Audit: rerun every example in the book and print book value against port value.

The book has 77 "Output from ... program" boxes. This program recomputes the
numbers in each box with the `finrecipes` package and prints one line per
number:

    ROW|box|chapter|label|book value as printed|port value
    NA|box|chapter|label|book value|reason

The Monte Carlo examples are run with the emulated C library rand()
(`random_seed_c_rand(1)` at the start of each box, as each example in the
book is a freshly started program), so they are compared digit for digit
like every other number.

`audit/make_table.py` turns these lines into docs/AUDIT.md.

Run with:  sh audit/run_audit.sh
"""

from std.math import exp, log, pow, sqrt

from finrecipes.cashflow import cash_flow_pv_discrete, cash_flow_irr_discrete
from finrecipes.bonds import (
    bonds_price_discrete,
    bonds_yield_to_maturity_discrete,
    bonds_duration_discrete,
    bonds_duration_macaulay_discrete,
    bonds_duration_modified_discrete,
    bonds_convexity_discrete,
)
from finrecipes.bonds import bonds_price as bonds_price_flat
from finrecipes.bonds import bonds_duration as bonds_duration_flat
from finrecipes.bonds import bonds_convexity as bonds_convexity_flat
from finrecipes.term_structure import (
    term_structure_discount_factor_from_yield,
    term_structure_yield_from_discount_factor,
    term_structure_forward_rate_from_discount_factors,
    term_structure_forward_rate_from_yields,
    term_structure_yield_linearly_interpolated,
    TermStructureFlat,
    TermStructureInterpolated,
    bonds_price,
    bonds_duration,
    bonds_convexity,
)
from finrecipes.matrix import mat_solve
from finrecipes.mean_variance import (
    mv_calculate_mean,
    mv_calculate_variance,
    mv_calculate_st_dev,
    mv_calculate_portfolio_given_mean_unconstrained,
    min_variance_frontier,
)
from finrecipes.futures import futures_price
from finrecipes.binomial_basic import (
    option_price_call_european_binomial_single_period,
    option_price_call_european_binomial_multi_period_given_ud,
)
from finrecipes.black_scholes import (
    option_price_call_black_scholes,
    option_price_put_black_scholes,
    option_price_delta_call_black_scholes,
    option_price_delta_put_black_scholes,
    option_price_partials_call_black_scholes,
    option_price_implied_volatility_call_black_scholes_newton,
    option_price_implied_volatility_call_black_scholes_bisections,
)
from finrecipes.warrants import warrant_price_adjusted_black_scholes
from finrecipes.black_scholes_extensions import (
    option_price_european_call_payout,
    option_price_european_call_dividends,
    option_price_american_call_one_dividend,
    futures_option_price_put_european_black,
    currency_option_price_call_european,
    option_price_american_perpetual_call,
)
from finrecipes.binomial import (
    option_price_call_european_binomial,
    option_price_call_american_binomial,
    option_price_put_american_binomial,
    option_price_partials_american_call_binomial,
    option_price_call_american_proportional_dividends_binomial,
    option_price_call_american_discrete_dividends_binomial,
    futures_option_price_call_american_binomial,
    currency_option_price_call_american_binomial,
)
from finrecipes.finite_differences import (
    option_price_put_european_finite_diff_explicit,
    option_price_put_american_finite_diff_explicit,
    option_price_put_european_finite_diff_implicit,
    option_price_put_american_finite_diff_implicit,
)
from finrecipes.random import random_seed_c_rand, random_uniform_0_1, random_normal
from finrecipes.simulation import (
    option_price_call_european_simulated,
    option_price_put_european_simulated,
    option_price_delta_call_european_simulated,
    option_price_delta_put_european_simulated,
    payoff_cash_or_nothing_call,
    payoff_asset_or_nothing_call,
)
from finrecipes.simulation import payoff_call as sim_payoff_call
from finrecipes.simulation import payoff_put as sim_payoff_put
from finrecipes.simulation import derivative_price_simulate_european_option_generic as sim_generic
from finrecipes.simulation import (
    derivative_price_simulate_european_option_generic_with_control_variate as sim_generic_cv,
)
from finrecipes.simulation import (
    derivative_price_simulate_european_option_generic_with_antithetic_variate as sim_generic_av,
)
from finrecipes.american_approximations import (
    option_price_american_put_approximated_johnson,
    option_price_american_call_approximated_baw,
)
from finrecipes.exotics import (
    option_price_put_bermudan_binomial,
    option_price_asian_geometric_average_price_call,
    option_price_european_lookback_call,
    option_price_european_lookback_put,
    payoff_arithmetric_average_call,
    payoff_geometric_average_call,
    payoff_lookback_call,
    payoff_lookback_put,
)
from finrecipes.exotics import derivative_price_simulate_european_option_generic as path_generic
from finrecipes.exotics import (
    derivative_price_simulate_european_option_generic_with_control_variate as path_generic_cv,
)
from finrecipes.generic_binomial import option_price_generic_binomial, payoff_binary_call
from finrecipes.generic_binomial import payoff_call as bin_payoff_call
from finrecipes.generic_binomial import payoff_put as bin_payoff_put
from finrecipes.trinomial import option_price_put_american_trinomial
from finrecipes.alternative_formulas import option_price_call_merton_jump_diffusion, heston_call_option_price
from finrecipes.bond_options import (
    bond_option_price_put_zero_black_scholes,
    bond_option_price_put_coupon_bond_black_scholes,
    bond_option_price_put_american_binomial,
)
from finrecipes.credit_risk import debt_value_merton
from finrecipes.term_structure_models import (
    term_structure_yield_nelson_siegel,
    TermStructureNelsonSiegel,
    term_structure_discount_factor_cubic_spline,
    TermStructureCubicSpline,
    term_structure_discount_factor_cir,
    TermStructureCir,
    term_structure_discount_factor_vasicek,
    TermStructureVasicek,
)
from finrecipes.rendleman_bartter import bond_option_price_call_zero_american_rendleman_bartter
from finrecipes.interest_rate_trees import (
    interest_rate_trees_gbm_build,
    interest_rate_trees_gbm_value_of_cashflows,
    interest_rate_trees_gbm_value_of_callable_bond,
)
from finrecipes.ho_lee import price_european_call_option_on_bond_using_ho_lee
from finrecipes.term_structure_derivatives import bond_option_price_call_zero_vasicek
from finrecipes.normdist import N, N2

def row(box: Int, chapter: String, label: String, book: String, port: Float64):
    """A deterministic number: the port should reproduce the printed digits."""
    print("ROW|" + String(box) + "|" + chapter + "|" + label + "|" + book + "|" + String(port))


def na(box: Int, chapter: String, label: String, book: String, reason: String):
    print("NA|" + String(box) + "|" + chapter + "|" + label + "|" + book + "|" + reason)


def discount(r: Float64, t: Float64) -> Float64:
    """Matlab d = 1/(1+r)^t, used by the Matlab boxes that print the discount vector."""
    return 1.0 / pow(1.0 + r, t)


def chapter_3():
    var times: List[Float64] = [0.0, 1.0, 2.0]
    var c1: List[Float64] = [-100.0, 75.0, 75.0]
    # box 1 (Matlab)
    row(1, "3", "d(1) at 10%", "1.0000", discount(0.1, 0.0))
    row(1, "3", "d(2) at 10%", "0.9091", discount(0.1, 1.0))
    row(1, "3", "d(3) at 10%", "0.8264", discount(0.1, 2.0))
    row(1, "3", "NPV", "30.165", cash_flow_pv_discrete(times, c1, 0.1))
    # box 2
    row(2, "3", "Present value, 10% discrete", "30.1653", cash_flow_pv_discrete(times, c1, 0.1))
    var c2: List[Float64] = [-100.0, 10.0, 110.0]
    # box 3 (Matlab)
    row(3, "3", "d(1) at 5%", "1.0000", discount(0.05, 0.0))
    row(3, "3", "d(2) at 5%", "0.9524", discount(0.05, 1.0))
    row(3, "3", "d(3) at 5%", "0.9070", discount(0.05, 2.0))
    row(3, "3", "NPV", "9.2971", cash_flow_pv_discrete(times, c2, 0.05))
    row(3, "3", "IRR", "0.100000", cash_flow_irr_discrete(times, c2))
    # box 4
    row(4, "3", "Present value, 5% discrete", "9.29705", cash_flow_pv_discrete(times, c2, 0.05))
    row(4, "3", "Internal rate of return", "0.1", cash_flow_irr_discrete(times, c2))
    # box 5 (Matlab, plain arithmetic; the book has no routine for it)
    var x: Float64 = 0.15 / 12.0
    var y: Float64 = 0.12 / 4.0
    row(5, "3", "r = 12 log(1+0.15/12) (inline arithmetic)", "0.1491", 12.0 * log(1.0 + x))
    row(5, "3", "r4 = 4 (exp(0.12/4)-1) (inline arithmetic)", "0.1218", 4.0 * (exp(y) - 1.0))


def chapter_4():
    var times: List[Float64] = [1.0, 2.0, 3.0]
    var cflows: List[Float64] = [10.0, 10.0, 110.0]
    var r: Float64 = 0.09
    var B = bonds_price_discrete(times, cflows, r)
    # box 6 (Matlab)
    row(6, "4", "d(1) at 9%", "0.9174", discount(r, 1.0))
    row(6, "4", "d(2) at 9%", "0.8417", discount(r, 2.0))
    row(6, "4", "d(3) at 9%", "0.7722", discount(r, 3.0))
    row(6, "4", "B", "102.53", B)
    # box 7
    row(7, "4", "bonds price", "102.531", B)
    # box 8 (Matlab)
    row(8, "4", "B", "102.53", B)
    row(8, "4", "y", "0.090000", bonds_yield_to_maturity_discrete(times, cflows, B))
    # box 9
    row(9, "4", "Bond price, 9% discrete", "102.531", B)
    row(9, "4", "bond yield to maturity", "0.09", bonds_yield_to_maturity_discrete(times, cflows, B))
    # box 10 (Matlab)
    row(10, "4", "B", "102.53", B)
    row(10, "4", "D", "2.7390", bonds_duration_discrete(times, cflows, r))
    row(10, "4", "y", "0.090000", bonds_yield_to_maturity_discrete(times, cflows, B))
    row(10, "4", "DM", "2.7390", bonds_duration_macaulay_discrete(times, cflows, B))
    # box 11
    row(11, "4", "bonds price", "102.531", B)
    row(11, "4", "bond duration", "2.73895", bonds_duration_discrete(times, cflows, r))
    row(11, "4", "bond macaulay", "2.73895", bonds_duration_macaulay_discrete(times, cflows, B))
    # box 12 (Matlab). The Matlab convexity is sum(t^2 C d)/B/(1+r)^2, which is
    # not the formula of C++ Code 4.6 (that one uses t(t+1)); computed inline.
    var cx: Float64 = 0.0
    for i in range(3):
        cx += times[i] * times[i] * cflows[i] * discount(r, times[i])
    cx = cx / B / ((1.0 + r) * (1.0 + r))
    row(12, "4", "B", "102.53", B)
    row(12, "4", "D", "2.7390", bonds_duration_discrete(times, cflows, r))
    row(12, "4", "Cx (Matlab formula, inline arithmetic)", "6.6272", cx)
    row(12, "4", "newB", "100.000", bonds_price_discrete(times, cflows, 0.1))
    # box 13
    row(13, "4", "bonds price", "102.531", B)
    row(13, "4", "bond duration", "2.73895", bonds_duration_discrete(times, cflows, r))
    row(13, "4", "bond duration modified", "2.5128", bonds_duration_modified_discrete(times, cflows, B))
    row(13, "4", "bond convexity", "8.93248", bonds_convexity_discrete(times, cflows, r))
    row(13, "4", "new bond price", "100", bonds_price_discrete(times, cflows, 0.1))
    # box 14 (Matlab)
    row(14, "4", "d(1) at 9%", "0.9174", discount(r, 1.0))
    row(14, "4", "d(2) at 9%", "0.8417", discount(r, 2.0))
    row(14, "4", "d(3) at 9%", "0.7722", discount(r, 3.0))
    row(14, "4", "B", "102.53", B)
    row(14, "4", "D", "2.7390", bonds_duration_discrete(times, cflows, r))
    row(14, "4", "Dadj", "2.5128", bonds_duration_modified_discrete(times, cflows, B))
    row(14, "4", "actualB", "100.000", bonds_price_discrete(times, cflows, 0.1))
    # box 15, continuous compounding
    row(15, "4", "bonds price", "101.464", bonds_price_flat(times, cflows, r))
    row(15, "4", "bond duration", "2.73753", bonds_duration_flat(times, cflows, r))
    row(15, "4", "bond convexity", "7.86779", bonds_convexity_flat(times, cflows, r))
    row(15, "4", "new bond price", "104.282", bonds_price_flat(times, cflows, 0.08))


def chapter_5() raises:
    # box 16
    var d_t1 = term_structure_discount_factor_from_yield(0.05, 1.0)
    var r_t2 = term_structure_yield_from_discount_factor(0.9, 2.0)
    row(16, "5", "discount factor from 1 period spot rate", "0.951229", d_t1)
    row(16, "5", "spot rate from 2 period discount factor", "0.0526803", r_t2)
    row(16, "5", "forward rate, from discount factors", "0.0553605", term_structure_forward_rate_from_discount_factors(d_t1, 0.9, 1.0))
    row(16, "5", "forward rate, from yields", "0.0553605", term_structure_forward_rate_from_yields(0.05, r_t2, 1.0, 2.0))
    # box 17
    var ts = TermStructureFlat(0.05)
    row(17, "5", "flat: discount factor t=1", "0.951229", ts.d(1.0))
    row(17, "5", "flat: discount factor t=2", "0.904837", ts.d(2.0))
    row(17, "5", "flat: spot rate t=1", "0.05", ts.r(1.0))
    row(17, "5", "flat: spot rate t=2", "0.05", ts.r(2.0))
    row(17, "5", "flat: forward rate 1 to 2", "0.05", ts.f(1.0, 2.0))
    # box 18 (Matlab): d = C^-1 B, r = d^(-1/t) - 1
    var C: List[List[Float64]] = [
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 110.0, 0.0, 0.0],
        [8.0, 8.0, 8.0, 8.0, 108.0, 0.0],
        [9.0, 9.0, 9.0, 9.0, 9.0, 109.0],
    ]
    var Bv: List[Float64] = [96.0, 94.0, 92.0, 118.0, 109.0, 112.0]
    var t: List[Float64] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    var dbook: List[String] = ["0.9600", "0.9400", "0.9200", "0.8164", "0.7399", "0.6662"]
    var rbook: List[String] = ["0.085069", "0.063830", "0.057162", "0.106772", "0.105628", "0.106884"]
    var d = mat_solve(C, Bv)
    for i in range(6):
        row(18, "5", "d(" + String(i + 1) + ")", dbook[i], d[i])
    for i in range(6):
        row(18, "5", "r(" + String(i + 1) + ") (inline arithmetic)", rbook[i], pow(d[i], -1.0 / t[i]) - 1.0)
    # box 19
    var times: List[Float64] = [0.1, 0.5, 1.0, 5.0, 10.0]
    var yields: List[Float64] = [0.1, 0.2, 0.3, 0.4, 0.5]
    row(19, "5", "interpolated yield t=0.1", "0.1", term_structure_yield_linearly_interpolated(0.1, times, yields))
    row(19, "5", "interpolated yield t=0.5", "0.2", term_structure_yield_linearly_interpolated(0.5, times, yields))
    row(19, "5", "interpolated yield t=1", "0.3", term_structure_yield_linearly_interpolated(1.0, times, yields))
    row(19, "5", "interpolated yield t=3", "0.35", term_structure_yield_linearly_interpolated(3.0, times, yields))
    row(19, "5", "interpolated yield t=5", "0.4", term_structure_yield_linearly_interpolated(5.0, times, yields))
    row(19, "5", "interpolated yield t=10", "0.5", term_structure_yield_linearly_interpolated(10.0, times, yields))
    # box 20
    var ti = TermStructureInterpolated([0.1, 1.0, 5.0], [0.05, 0.07, 0.08])
    row(20, "5", "interpolated: discount factor t=1", "0.932394", ti.d(1.0))
    row(20, "5", "interpolated: discount factor t=2", "0.865022", ti.d(2.0))
    row(20, "5", "interpolated: spot rate t=1", "0.07", ti.r(1.0))
    row(20, "5", "interpolated: spot rate t=2", "0.0725", ti.r(2.0))
    row(20, "5", "interpolated: forward rate 1 to 2", "0.075", ti.f(1.0, 2.0))
    # box 21
    var bt: List[Float64] = [1.0, 2.0]
    var bc: List[Float64] = [10.0, 110.0]
    var tsflat = TermStructureFlat(0.1)
    row(21, "5", "price", "99.1088", bonds_price(bt, bc, tsflat))
    row(21, "5", "duration", "1.9087", bonds_duration(bt, bc, tsflat))
    row(21, "5", "convexity", "3.72611", bonds_convexity(bt, bc, tsflat))


def chapter_6() raises:
    var e: List[Float64] = [0.1, 0.11, 0.08]
    var V: List[List[Float64]] = [[0.2, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.15]]
    var third: Float64 = 1.0 / 3.0
    var w: List[Float64] = [third, third, third]
    # box 22 (Matlab)
    row(22, "6", "er", "0.096667", mv_calculate_mean(e, w))
    row(22, "6", "sigma", "0.2236", mv_calculate_st_dev(V, w))
    # box 23 (Matlab)
    var fr = min_variance_frontier(e, V)
    row(23, "6", "a", "2.1333", fr.a)
    row(23, "6", "b", "0.2137", fr.b)
    row(23, "6", "c", "21.667", fr.c)
    row(23, "6", "d", "0.078333", fr.d)
    var gbook: List[String] = ["0.021277", "-2.680851", "3.659574"]
    var hbook: List[String] = ["2.1277", "31.9149", "-34.0426"]
    var wbook: List[String] = ["0.2128", "0.1915", "0.5957"]
    var wp = fr.weights(0.09)
    for i in range(3):
        row(23, "6", "g(" + String(i + 1) + ")", gbook[i], fr.g[i])
    for i in range(3):
        row(23, "6", "h(" + String(i + 1) + ")", hbook[i], fr.h[i])
    for i in range(3):
        row(23, "6", "w(" + String(i + 1) + ")", wbook[i], wp[i])
    # box 24
    var e2: List[Float64] = [0.05, 0.1]
    var V2: List[List[Float64]] = [[1.0, 0.0], [0.0, 1.0]]
    var w2: List[Float64] = [0.5, 0.5]
    row(24, "6", "mean", "0.075", mv_calculate_mean(e2, w2))
    row(24, "6", "variance", "0.5", mv_calculate_variance(V2, w2))
    row(24, "6", "stdev", "0.707107", mv_calculate_st_dev(V2, w2))
    # box 25
    var w3 = mv_calculate_portfolio_given_mean_unconstrained(e2, V2, 0.075)
    row(25, "6", "w1", "0.5", w3[0])
    row(25, "6", "w2", "0.5", w3[1])


def chapters_7_to_11():
    # box 26
    row(26, "7", "futures price", "105.127", futures_price(100.0, 0.10, 0.5))
    # box 27
    var u: Float64 = 1.05
    var d = 1.0 / u
    row(27, "8", "one period european call", "3.64342", option_price_call_european_binomial_single_period(100.0, 100.0, 0.025, u, d))
    row(27, "8", "two period european call", "5.44255", option_price_call_european_binomial_multi_period_given_ud(100.0, 100.0, 0.025, u, d, 2))
    # box 28 (Matlab)
    row(28, "9", "c", "10.308", option_price_call_black_scholes(100.0, 100.0, 0.1, 0.1, 1.0))
    # box 29
    row(29, "9", "Black Scholes call price", "5.45325", option_price_call_black_scholes(50.0, 50.0, 0.10, 0.30, 0.50))
    # box 30
    var g = option_price_partials_call_black_scholes(50.0, 50.0, 0.10, 0.30, 0.50)
    row(30, "9", "Delta", "0.633737", g.delta)
    row(30, "9", "Gamma", "0.0354789", g.gamma)
    row(30, "9", "Theta", "-6.61473", g.theta)
    row(30, "9", "Vega", "13.3046", g.vega)
    row(30, "9", "Rho", "13.1168", g.rho)
    # box 31
    row(31, "9", "implied volatility, Newton", "0.0500427", option_price_implied_volatility_call_black_scholes_newton(50.0, 50.0, 0.10, 0.50, 2.5))
    row(31, "9", "implied volatility, bisections", "0.0500419", option_price_implied_volatility_call_black_scholes_bisections(50.0, 50.0, 0.10, 0.50, 2.5))
    # box 32
    row(32, "10", "warrant price", "10.142", warrant_price_adjusted_black_scholes(48.0, 40.0, 0.08, 0.30, 0.5, 1000.0, 10000.0))
    # box 33
    var dt: List[Float64] = [0.25, 0.75]
    var da: List[Float64] = [2.5, 2.5]
    row(33, "11", "european call, continuous dividend", "11.7344", option_price_european_call_payout(100.0, 100.0, 0.1, 0.05, 0.25, 1.0))
    row(33, "11", "european call, discrete dividend", "11.8094", option_price_european_call_dividends(100.0, 100.0, 0.1, 0.25, 1.0, dt, da))
    # box 34
    row(34, "11", "american call with one dividend", "10.0166", option_price_american_call_one_dividend(100.0, 100.0, 0.1, 0.25, 1.0, 10.0, 0.5))
    # box 35: labelled "call" in the book, but the program calls the put routine
    row(35, "11", "european futures option (book label: call; program prices the put)", "0.851476", futures_option_price_put_european_black(50.0, 45.0, 0.08, 0.2, 0.5))
    # box 36
    row(36, "11", "european currency call option", "2.22556", currency_option_price_call_european(50.0, 52.0, 0.08, 0.05, 0.2, 0.5))
    # box 37
    row(37, "11", "perpetual call price", "19.4767", option_price_american_perpetual_call(50.0, 40.0, 0.05, 0.02, 0.05))


def chapters_12_13() raises:
    var S: Float64 = 100.0
    var K: Float64 = 100.0
    var r: Float64 = 0.1
    var sigma: Float64 = 0.25
    var time: Float64 = 1.0
    # box 38
    row(38, "12", "european call", "14.9505", option_price_call_european_binomial(S, K, r, sigma, time, 100))
    row(38, "12", "american call", "14.9505", option_price_call_american_binomial(S, K, r, sigma, time, 100))
    row(38, "12", "american put", "6.54691", option_price_put_american_binomial(S, K, r, sigma, time, 100))
    # box 39 (Matlab)
    row(39, "12", "C", "14.951", option_price_call_american_binomial(S, K, r, sigma, time, 100))
    row(39, "12", "P", "6.5469", option_price_put_american_binomial(S, K, r, sigma, time, 100))
    # box 40
    var g = option_price_partials_american_call_binomial(S, K, r, sigma, time, 100)
    row(40, "12", "delta", "0.699792", g.delta)
    row(40, "12", "gamma", "0.0140407", g.gamma)
    row(40, "12", "theta", "-9.89067", g.theta)
    row(40, "12", "vega", "34.8536", g.vega)
    row(40, "12", "rho", "56.9652", g.rho)
    # box 41
    var dt: List[Float64] = [0.25, 0.75]
    var dy: List[Float64] = [0.025, 0.025]
    var da: List[Float64] = [2.5, 2.5]
    row(41, "12", "call, continuous dividend payout", "13.5926", option_price_call_american_binomial(S, K, r, 0.02, sigma, time, 100))
    row(41, "12", "call, proportional dividend yields", "11.8604", option_price_call_american_proportional_dividends_binomial(S, K, r, sigma, time, 100, dt, dy))
    row(41, "12", "call, discrete dividend amounts", "12.0233", option_price_call_american_discrete_dividends_binomial(S, K, r, sigma, time, 100, dt, da))
    # box 42
    row(42, "12", "futures call option, binomial", "5.74254", futures_option_price_call_american_binomial(50.0, 45.0, 0.08, 0.2, 0.5, 100))
    # box 43
    row(43, "12", "currency call option, binomial", "2.23129", currency_option_price_call_american_binomial(50.0, 52.0, 0.08, 0.05, 0.2, 0.5, 100))
    # box 44
    row(44, "13", "explicit finite differences, european put", "4.03667", option_price_put_european_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 20, 11))
    row(44, "13", "explicit finite differences, american put", "4.25085", option_price_put_american_finite_diff_explicit(50.0, 50.0, 0.1, 0.4, 0.4167, 20, 11))
    # box 45
    row(45, "13", "black scholes put price", "4.35166", option_price_put_black_scholes(50.0, 50.0, 0.1, 0.4, 0.5))
    row(45, "13", "implicit Euro put price", "4.34731", option_price_put_european_finite_diff_implicit(50.0, 50.0, 0.1, 0.4, 0.5, 200, 200))
    row(45, "13", "implicit American put price", "4.60064", option_price_put_american_finite_diff_implicit(50.0, 50.0, 0.1, 0.4, 0.5, 200, 200))
    # box 46 (Matlab)
    row(46, "13", "P", "4.6006", option_price_put_american_finite_diff_implicit(50.0, 50.0, 0.1, 0.4, 0.5, 200, 200))


def chapter_14():
    var S: Float64 = 100.0
    var K: Float64 = 100.0
    var r: Float64 = 0.1
    var sigma: Float64 = 0.25
    var time: Float64 = 1.0
    # box 47, 5000 simulations
    random_seed_c_rand(1)
    row(47, "14", "call: black scholes price", "14.9758", option_price_call_black_scholes(S, K, r, sigma, time))
    row(47, "14", "call: simulated price", "14.8404", option_price_call_european_simulated(S, K, r, sigma, time, 5000))
    row(47, "14", "put: black scholes price", "5.45954", option_price_put_black_scholes(S, K, r, sigma, time))
    row(47, "14", "put: simulated price", "5.74588", option_price_put_european_simulated(S, K, r, sigma, time, 5000))
    # box 48, 5000 simulations
    random_seed_c_rand(1)
    row(48, "14", "call: bs delta", "0.700208", option_price_delta_call_black_scholes(S, K, r, sigma, time))
    row(48, "14", "call: sim delta", "0.701484", option_price_delta_call_european_simulated(S, K, r, sigma, time, 5000))
    row(48, "14", "put: bs delta", "-0.299792", option_price_delta_put_black_scholes(S, K, r, sigma, time))
    row(48, "14", "put: sim delta", "-0.307211", option_price_delta_put_european_simulated(S, K, r, sigma, time, 5000))
    # box 49, 50000 simulations
    random_seed_c_rand(1)
    row(49, "14", "Black Scholes call option price", "14.9758", option_price_call_black_scholes(S, K, r, sigma, time))
    row(49, "14", "Simulated call option price", "14.995", sim_generic(S, K, r, sigma, time, sim_payoff_call, 50000))
    row(49, "14", "Black Scholes put option price", "5.45954", option_price_put_black_scholes(S, K, r, sigma, time))
    row(49, "14", "Simulated put option price", "5.5599", sim_generic(S, K, r, sigma, time, sim_payoff_put, 50000))
    # box 50, 50000 simulations
    random_seed_c_rand(1)
    row(50, "14", "Black Scholes call option price", "14.9758", option_price_call_black_scholes(S, K, r, sigma, time))
    row(50, "14", "Simulated call option price", "14.995", sim_generic(S, K, r, sigma, time, sim_payoff_call, 50000))
    row(50, "14", "Simulated call option price, CV", "14.9758", sim_generic_cv(S, K, r, sigma, time, sim_payoff_call, 50000))
    row(50, "14", "Simulated call option price, AV", "14.9919", sim_generic_av(S, K, r, sigma, time, sim_payoff_call, 50000))
    row(50, "14", "Black Scholes put option price", "5.45954", option_price_put_black_scholes(S, K, r, sigma, time))
    row(50, "14", "Simulated put option price", "5.41861", sim_generic(S, K, r, sigma, time, sim_payoff_put, 50000))
    row(50, "14", "Simulated put option price, CV", "5.42541", sim_generic_cv(S, K, r, sigma, time, sim_payoff_put, 50000))
    row(50, "14", "Simulated put option price, AV", "5.46043", sim_generic_av(S, K, r, sigma, time, sim_payoff_put, 50000))
    # box 51, binary options, 5000 simulations
    random_seed_c_rand(1)
    row(51, "14", "cash or nothing, Q=1", "0.547427", sim_generic(S, K, r, sigma, time, payoff_cash_or_nothing_call, 5000))
    row(51, "14", "cash or nothing, control variate", "1.02552", sim_generic_cv(S, K, r, sigma, time, payoff_cash_or_nothing_call, 5000))
    row(51, "14", "cash or nothing, antithetic variate", "0.549598", sim_generic_av(S, K, r, sigma, time, payoff_cash_or_nothing_call, 5000))
    row(51, "14", "asset or nothing", "70.5292", sim_generic(S, K, r, sigma, time, payoff_asset_or_nothing_call, 5000))
    row(51, "14", "asset or nothing, control variate", "69.8451", sim_generic_cv(S, K, r, sigma, time, payoff_asset_or_nothing_call, 5000))
    row(51, "14", "asset or nothing, antithetic variate", "70.2205", sim_generic_av(S, K, r, sigma, time, payoff_asset_or_nothing_call, 5000))


def chapters_15_16():
    # box 52
    row(52, "15", "American put, Johnson approximation", "0.11582", option_price_american_put_approximated_johnson(1.1, 1.0, 0.125, 0.5, 1.0))
    # box 53
    row(53, "15", "Call price, Barone-Adesi Whaley (listing as printed)", "5.74339", option_price_american_call_approximated_baw(100.0, 100.0, 0.08, -0.04, 0.20, 0.25))
    # box 54
    var ex: List[Float64] = [0.25, 0.5, 0.75]
    row(54, "16", "Bermudan put price", "15.8869", option_price_put_bermudan_binomial(80.0, 100.0, 0.20, 0.0, 0.25, 1.0, ex, 500))
    # box 55
    row(55, "16", "Analytical geometric average", "6.74876", option_price_asian_geometric_average_price_call(100.0, 100.0, 0.06, 0.0, 0.25, 1.0))
    # box 56
    row(56, "16", "Lookback call price", "27.0713", option_price_european_lookback_call(100.0, 100.0, 0.06, 0.0, 0.346, 1.0))
    # box 57, 10000 simulations of 250 steps
    var S: Float64 = 100.0
    var r: Float64 = 0.10
    var sigma: Float64 = 0.25
    var time: Float64 = 1.0
    random_seed_c_rand(1)
    row(57, "16", "simulated arithmetic average", "1.49696", path_generic(S, 120.0, r, sigma, time, payoff_arithmetric_average_call, 250, 10000))
    row(57, "16", "simulated geometric average", "1.38017", path_generic(S, 120.0, r, sigma, time, payoff_geometric_average_call, 250, 10000))
    row(57, "16", "analytical lookback put", "16.2665", option_price_european_lookback_put(S, S, r, 0.0, sigma, time))
    row(57, "16", "simulated lookback put", "14.9846", path_generic(S, 0.0, r, sigma, time, payoff_lookback_put, 250, 10000))
    row(57, "16", "analytical lookback call", "22.8089", option_price_european_lookback_call(S, S, r, 0.0, sigma, time))
    row(57, "16", "simulated lookback call", "21.9336", path_generic(S, 0.0, r, sigma, time, payoff_lookback_call, 250, 10000))
    row(57, "16", "simulated lookback call, control variates", "22.0685", path_generic_cv(S, 0.0, r, sigma, time, payoff_lookback_call, 250, 10000))


def chapters_17_to_21():
    # box 58
    row(58, "17", "american call price", "14.9505", option_price_generic_binomial(100.0, 100.0, bin_payoff_call, 0.1, 0.25, 1.0, 100))
    row(58, "17", "american put price", "6.54691", option_price_generic_binomial(100.0, 100.0, bin_payoff_put, 0.1, 0.25, 1.0, 100))
    # box 59
    row(59, "17", "binary option price", "0.498858", option_price_generic_binomial(100.0, 120.0, payoff_binary_call, 0.1, 0.25, 1.0, 100))
    # box 60
    row(60, "18", "american put, trinomial", "6.52719", option_price_put_american_trinomial(100.0, 100.0, 0.1, 0.0, 0.25, 1.0, 100))
    # box 61
    row(61, "19", "Merton jump diffusion call", "23.2074", option_price_call_merton_jump_diffusion(100.0, 100.0, 0.05, 0.3, 1.0, 0.5, 0.5, 0.5))
    # box 62
    row(62, "19", "heston call price", "3.06944", heston_call_option_price(100.0, 100.0, 0.01, 0.01, 0.5, 0.0, 2.0, 0.0, 0.01, 0.01))
    # box 63
    var ct: List[Float64] = [0.5]
    var cc: List[Float64] = [1.0]
    row(63, "20", "zero coupon put option price", "1.92791", bond_option_price_put_zero_black_scholes(100.0, 100.0, 0.05, 0.1, 1.0))
    row(63, "20", "coupon bond put option price", "2.22852", bond_option_price_put_coupon_bond_black_scholes(100.0, 100.0, 0.05, 0.1, 1.0, ct, cc))
    row(63, "20", "zero coupon american put, binomial", "2.43282", bond_option_price_put_american_binomial(100.0, 100.0, 0.05, 0.1, 1.0, 100))
    # box 64
    row(64, "21", "Debt value", "81.8592", debt_value_merton(100.0, 90.0, 0.05, 0.25, 1.0))


def chapters_22_to_26():
    # box 65
    var ns = TermStructureNelsonSiegel(0.01, 0.01, 0.01, 5.0)
    row(65, "22", "Nelson Siegel: direct calculation, yield", "0.0363142", term_structure_yield_nelson_siegel(1.0, 0.01, 0.01, 0.01, 5.0))
    row(65, "22", "Nelson Siegel: yield (t=1)", "0.0363142", ns.r(1.0))
    row(65, "22", "Nelson Siegel: discount factor (t=1)", "0.964337", ns.d(1.0))
    row(65, "22", "Nelson Siegel: forward rate (1, 2)", "0.0300602", ns.f(1.0, 2.0))
    # box 66
    var fs: List[Float64] = [0.01, 0.01, -0.01]
    var knots: List[Float64] = [2.0, 7.0, 12.0]
    var cs = TermStructureCubicSpline(0.1, 0.1, -0.1, fs, knots)
    row(66, "22", "cubic spline: direct, discount factor (t=1)", "1.1", term_structure_discount_factor_cubic_spline(1.0, 0.1, 0.1, -0.1, fs, knots))
    row(66, "22", "cubic spline: yield (t=1)", "-0.0953102", cs.r(1.0))
    row(66, "22", "cubic spline: discount factor (t=1)", "1.1", cs.d(1.0))
    row(66, "22", "cubic spline: forward (1, 2)", "0.318454", cs.f(1.0, 2.0))
    # box 67
    var cir = TermStructureCir(0.05, 0.01, 0.0, 0.08, 0.1)
    row(67, "22", "CIR: direct, discount factor (t=1)", "0.951166", term_structure_discount_factor_cir(1.0, 0.05, 0.01, 0.0, 0.08, 0.1))
    row(67, "22", "CIR: yield (t=1)", "0.0500668", cir.r(1.0))
    row(67, "22", "CIR: discount factor (t=1)", "0.951166", cir.d(1.0))
    row(67, "22", "CIR: forward (1, 2)", "0.0498756", cir.f(1.0, 2.0))
    # box 68
    var vc = TermStructureVasicek(0.05, -0.1, 0.1, 0.1)
    row(68, "22", "Vasicek: direct, discount factor (t=1)", "0.955408", term_structure_discount_factor_vasicek(1.0, 0.05, -0.1, 0.1, 0.1))
    row(68, "22", "Vasicek: yield (t=1)", "0.0456168", vc.r(1.0))
    row(68, "22", "Vasicek: discount factor (t=1)", "0.955408", vc.d(1.0))
    row(68, "22", "Vasicek: forward rate (1, 2)", "0.0281476", vc.f(1.0, 2.0))
    # box 69
    row(69, "23", "Rendleman Bartter option on zero coupon bond", "0.00713661", bond_option_price_call_zero_american_rendleman_bartter(950.0, 4.0, 0.15, 0.05, 0.10, 5.0, 1000.0, 100))
    # box 70
    var tree = interest_rate_trees_gbm_build(0.1, 1.02, 0.99, 3)
    row(70, "24", "tree time 0", "0.1", tree[0][0])
    row(70, "24", "tree time 1, node 0", "0.099", tree[1][0])
    row(70, "24", "tree time 1, node 1", "0.102", tree[1][1])
    row(70, "24", "tree time 2, node 0", "0.09801", tree[2][0])
    row(70, "24", "tree time 2, node 1", "0.10098", tree[2][1])
    row(70, "24", "tree time 2, node 2", "0.10404", tree[2][2])
    # box 71
    var cf: List[Float64] = [0.0, 10.0, 10.0, 110.0]
    row(71, "24", "Bond price B", "98.5997", interest_rate_trees_gbm_value_of_cashflows(cf, tree, 0.5))
    # box 72
    var tree2 = interest_rate_trees_gbm_build(0.06, 1.2, 0.9, 10)
    var cf2: List[Float64] = [0.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 106.0]
    row(72, "24", "Straight bond price", "89.9017", interest_rate_trees_gbm_value_of_cashflows(cf2, tree2, 0.5))
    row(72, "24", "Callable bond price", "89.2483", interest_rate_trees_gbm_value_of_callable_bond(cf2, tree2, 0.5, 6, 106.0))
    # box 73
    var times: List[Float64] = [5.0]
    var cflows: List[Float64] = [100.0]
    row(73, "25", "Ho-Lee call, flat term structure", "6.4857", price_european_call_option_on_bond_using_ho_lee(TermStructureFlat(0.1), 0.98, 0.5, times, cflows, 80.0, 3.0))
    row(73, "25", "Ho-Lee call, Nelson Siegel term structure", "6.35411", price_european_call_option_on_bond_using_ho_lee(TermStructureNelsonSiegel(0.09, 0.01, 0.01, 5.0), 0.98, 0.5, times, cflows, 80.0, 3.0))
    # box 74
    row(74, "26", "Vasicek call option price", "0.000226833", bond_option_price_call_zero_vasicek(0.9, 0.05, 1.0, 5.0, 0.1, 0.1, 0.02))


def chapter_27_and_appendix():
    # box 75
    na(75, "27", "Boost date and time demo (4 printed lines)", "date: 2014-Apr-01 ...", "Demonstrates the Boost C++ library and prints the current clock time; not ported")
    # box 76
    row(76, "A", "N(0)", "0.5", N(0.0))
    row(76, "A", "N(0,0,0)", "0.25", N2(0.0, 0.0, 0.0))
    # box 77: the first numbers of the C library rand() sequence
    random_seed_c_rand(1)
    var ub: List[String] = ["0.840188", "0.394383", "0.783099", "0.79844", "0.911647"]
    var nb: List[String] = ["-1.07224", "0.925946", "2.70202", "1.36918", "0.0187313"]
    for i in range(5):
        row(77, "A", "random uniform number " + String(i + 1), ub[i], random_uniform_0_1())
    for i in range(5):
        row(77, "A", "random normal number " + String(i + 1), nb[i], random_normal())


def main() raises:
    chapter_3()
    chapter_4()
    chapter_5()
    chapter_6()
    chapters_7_to_11()
    chapters_12_13()
    chapter_14()
    chapters_15_16()
    chapters_17_to_21()
    chapters_22_to_26()
    chapter_27_and_appendix()
