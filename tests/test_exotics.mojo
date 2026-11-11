from std.math import exp, log, sqrt
from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_true
from finrecipes.black_scholes import option_price_call_black_scholes, option_price_put_black_scholes
from finrecipes.normdist import N
from finrecipes.random import random_seed
from finrecipes.exotics import (
    option_price_put_bermudan_binomial,
    option_price_call_bermudan_binomial,
    option_price_asian_geometric_average_price_call,
    option_price_european_lookback_call,
    option_price_european_lookback_put,
    simulate_lognormally_distributed_sequence,
    derivative_price_simulate_european_option_generic,
    derivative_price_simulate_european_option_generic_with_control_variate,
    payoff_arithmetric_average_call,
    payoff_geometric_average_call,
    payoff_lookback_call,
    payoff_lookback_put,
)

# The simulation example of section 16.5 uses S=100, K=120, r=0.10,
# sigma=0.25, time=1, 250 steps and 10000 simulations.
#
# The book's simulated numbers come from the C library rand() and cannot be
# reproduced digit for digit. Tolerances for the Monte Carlo tests are 4
# standard errors, standard error = (standard deviation of one discounted
# payoff) / sqrt(no_sims). The standard deviations quoted in the comments
# were measured in a separate run of 200,000 paths. Where the only reference
# is the book's own simulated value, that value has the same standard error
# as ours, so the tolerance is 4 * sqrt(2) standard errors. The generator is
# seeded, so the runs are reproducible.


def four_standard_errors(sd: Float64, no_sims: Int) -> Float64:
    return 4.0 * sd / sqrt(Float64(no_sims))


def payoff_terminal_call(prices: List[Float64], K: Float64) -> Float64:
    """A payoff on the path that only looks at the terminal price."""
    return max(0.0, prices[len(prices) - 1] - K)


def test_bermudan_put_book_example() raises:
    # C++ Code 16.1. Book output: Bermudan put price = 15.8869
    var potential_exercise_times: List[Float64] = [0.25, 0.5, 0.75]
    var p = option_price_put_bermudan_binomial(80.0, 100.0, 0.20, 0.0, 0.25, 1.0, potential_exercise_times, 500)
    assert_almost_equal(p, 15.8869, atol=1e-4)


def test_bermudan_put_between_european_and_american() raises:
    var no_exercise = List[Float64]()
    var three_dates: List[Float64] = [0.25, 0.5, 0.75]
    var every_step = List[Float64]()
    for i in range(1, 500):
        # just past each step so that truncation to a step number is exact
        every_step.append((Float64(i) + 0.5) / 500.0)
    var european = option_price_put_bermudan_binomial(80.0, 100.0, 0.20, 0.0, 0.25, 1.0, no_exercise, 500)
    var bermudan = option_price_put_bermudan_binomial(80.0, 100.0, 0.20, 0.0, 0.25, 1.0, three_dates, 500)
    var american = option_price_put_bermudan_binomial(80.0, 100.0, 0.20, 0.0, 0.25, 1.0, every_step, 500)
    # without early exercise dates the tree prices the European put
    assert_almost_equal(european, option_price_put_black_scholes(80.0, 100.0, 0.20, 0.25, 1.0), atol=0.01)
    assert_true(european < bermudan)
    assert_true(bermudan < american)
    # exercise at every step before maturity, but not today: the option is
    # deep in the money and worth slightly less than immediate exercise
    assert_true(american > 19.5 and american <= 20.0)
    # exercise times outside (0, time) are ignored
    var outside: List[Float64] = [0.0, 1.0, 1.5, -0.5]
    assert_almost_equal(
        option_price_put_bermudan_binomial(80.0, 100.0, 0.20, 0.0, 0.25, 1.0, outside, 500), european, atol=1e-12
    )


def test_bermudan_call() raises:
    var no_exercise = List[Float64]()
    var three_dates: List[Float64] = [0.25, 0.5, 0.75]
    # no payout: early exercise of a call is never optimal, the Bermudan call
    # is the European call
    var european = option_price_call_bermudan_binomial(100.0, 100.0, 0.10, 0.0, 0.25, 1.0, no_exercise, 500)
    var bermudan = option_price_call_bermudan_binomial(100.0, 100.0, 0.10, 0.0, 0.25, 1.0, three_dates, 500)
    assert_almost_equal(european, option_price_call_black_scholes(100.0, 100.0, 0.10, 0.25, 1.0), atol=0.02)
    assert_almost_equal(bermudan, european, atol=1e-10)
    # with a payout early exercise has value
    european = option_price_call_bermudan_binomial(120.0, 100.0, 0.10, 0.12, 0.25, 1.0, no_exercise, 500)
    bermudan = option_price_call_bermudan_binomial(120.0, 100.0, 0.10, 0.12, 0.25, 1.0, three_dates, 500)
    assert_true(bermudan > european + 0.1)
    # put-call parity of the European values on the same tree
    var put = option_price_put_bermudan_binomial(120.0, 100.0, 0.10, 0.12, 0.25, 1.0, no_exercise, 500)
    assert_almost_equal(european - put, 120.0 * exp(Float64(-0.12)) - 100.0 * exp(Float64(-0.10)), atol=1e-8)


def test_asian_geometric_average_price_call() raises:
    # C++ Code 16.2. Book output: Analytical geometric average = 6.74876
    var c = option_price_asian_geometric_average_price_call(100.0, 100.0, 0.06, 0.0, 0.25, 1.0)
    assert_almost_equal(c, 6.74876, atol=1e-5)
    # an average price call is worth less than the ordinary call
    assert_true(c < option_price_call_black_scholes(100.0, 100.0, 0.06, 0.25, 1.0))


def test_lookback_analytical() raises:
    # C++ Code 16.3. Book output: Lookback call price = 27.0713
    assert_almost_equal(option_price_european_lookback_call(100.0, 100.0, 0.06, 0.0, 0.346, 1.0), 27.0713, atol=1e-4)
    # Example of section 16.5. Book output: analytical lookback put = 16.2665,
    # analytical lookback call = 22.8089
    assert_almost_equal(option_price_european_lookback_put(100.0, 100.0, 0.10, 0.0, 0.25, 1.0), 16.2665, atol=1e-4)
    assert_almost_equal(option_price_european_lookback_call(100.0, 100.0, 0.10, 0.0, 0.25, 1.0), 22.8089, atol=1e-4)
    # the listing returns 0 when r equals q
    assert_almost_equal(option_price_european_lookback_call(100.0, 100.0, 0.06, 0.06, 0.25, 1.0), 0.0, atol=1e-12)
    assert_almost_equal(option_price_european_lookback_put(100.0, 100.0, 0.06, 0.06, 0.25, 1.0), 0.0, atol=1e-12)
    # a lookback call is worth more than the at the money call, and more
    # still when the minimum so far is below the current price
    var c = option_price_european_lookback_call(100.0, 100.0, 0.10, 0.0, 0.25, 1.0)
    assert_true(c > option_price_call_black_scholes(100.0, 100.0, 0.10, 0.25, 1.0))
    assert_true(option_price_european_lookback_call(100.0, 90.0, 0.10, 0.0, 0.25, 1.0) > c)
    var p = option_price_european_lookback_put(100.0, 100.0, 0.10, 0.0, 0.25, 1.0)
    assert_true(p > option_price_put_black_scholes(100.0, 100.0, 0.10, 0.25, 1.0))
    assert_true(option_price_european_lookback_put(100.0, 110.0, 0.10, 0.0, 0.25, 1.0) > p)


def test_payoff_functions() raises:
    # C++ Code 16.6 and 16.7
    var prices: List[Float64] = [100.0, 110.0, 90.0, 120.0]
    assert_almost_equal(payoff_arithmetric_average_call(prices, 100.0), 5.0, atol=1e-12)
    assert_almost_equal(payoff_arithmetric_average_call(prices, 110.0), 0.0, atol=1e-12)
    var geometric_average = sqrt(sqrt(Float64(100.0) * 110.0 * 90.0 * 120.0))  # 104.4009
    assert_almost_equal(geometric_average, 104.4009, atol=1e-4)
    assert_almost_equal(payoff_geometric_average_call(prices, 100.0), geometric_average - 100.0, atol=1e-10)
    assert_almost_equal(payoff_geometric_average_call(prices, 110.0), 0.0, atol=1e-12)
    assert_almost_equal(payoff_lookback_call(prices, 0.0), 30.0, atol=1e-12)
    assert_almost_equal(payoff_lookback_put(prices, 0.0), 0.0, atol=1e-12)
    var prices2: List[Float64] = [100.0, 130.0, 95.0, 105.0]
    assert_almost_equal(payoff_lookback_call(prices2, 0.0), 10.0, atol=1e-12)
    assert_almost_equal(payoff_lookback_put(prices2, 0.0), 25.0, atol=1e-12)


def test_simulate_lognormally_distributed_sequence() raises:
    # C++ Code 16.4. The terminal price of a path is lognormal whatever the
    # number of steps: mean S exp(r T) = 110.517 with standard deviation
    # 28.067, and log(S_T) has mean log(S) + (r - sigma^2/2) T with standard
    # deviation 0.25.
    random_seed(164)
    var no_sims = 20000
    var no_steps = 50
    var sum: Float64 = 0.0
    var sum_log: Float64 = 0.0
    for _ in range(no_sims):
        var prices = simulate_lognormally_distributed_sequence(100.0, 0.10, 0.25, 1.0, no_steps)
        assert_equal(len(prices), no_steps)
        for i in range(no_steps):
            assert_true(prices[i] > 0.0)
        sum += prices[no_steps - 1]
        sum_log += log(prices[no_steps - 1])
    assert_almost_equal(sum / Float64(no_sims), 100.0 * exp(Float64(0.10)), atol=four_standard_errors(28.067, no_sims))
    assert_almost_equal(
        sum_log / Float64(no_sims),
        log(Float64(100.0)) + (0.10 - 0.5 * 0.25 * 0.25),
        atol=four_standard_errors(0.25, no_sims),
    )


def test_simulated_asian_options() raises:
    # Example of section 16.5, C++ Code 16.5 with the payoffs of C++ Code 16.6.
    # Book output: simulated arithmetric average price=1.49696,
    # simulated geometric average = 1.38017.
    var S: Float64 = 100.0
    var K: Float64 = 120.0
    var r: Float64 = 0.10
    var time: Float64 = 1.0
    var sigma: Float64 = 0.25
    var no_sims = 10000
    var no_steps = 250
    # The geometric average of the 250 simulated prices is lognormal, so the
    # expectation of the simulated geometric average call is known exactly:
    # log of the average has mean m and variance v below. (It differs a
    # little from C++ Code 16.2, which is for continuous sampling: 1.3303.)
    var nf = Float64(no_steps)
    var delta_t = time / nf
    var m = log(S) + (r - 0.5 * sigma * sigma) * delta_t * (nf + 1.0) / 2.0
    var v = sigma * sigma * delta_t * (nf + 1.0) * (2.0 * nf + 1.0) / (6.0 * nf)
    var d2 = (m - log(K)) / sqrt(v)
    var d1 = d2 + sqrt(v)
    var geometric_exact = exp(-r * time) * (exp(m + 0.5 * v) * N(d1) - K * N(d2))
    assert_almost_equal(geometric_exact, 1.34499, atol=1e-5)
    assert_almost_equal(
        option_price_asian_geometric_average_price_call(S, K, r, 0.0, sigma, time), 1.3303, atol=1e-4
    )
    # One discounted geometric average payoff has standard deviation 4.40:
    # 4 standard errors are 0.176. The book's 1.38017 is within that too.
    random_seed(165)
    var geometric = derivative_price_simulate_european_option_generic(
        S, K, r, sigma, time, payoff_geometric_average_call, no_steps, no_sims
    )
    assert_almost_equal(geometric, geometric_exact, atol=four_standard_errors(4.40, no_sims))
    # The arithmetic average has no closed form; the reference is the book's
    # simulated 1.49696. One discounted payoff has standard deviation 4.89,
    # so 4 * sqrt(2) standard errors are 0.277. (A 200,000 path run gives
    # 1.546 with standard error 0.011.)
    random_seed(165)  # the same paths as for the geometric average
    var arithmetic = derivative_price_simulate_european_option_generic(
        S, K, r, sigma, time, payoff_arithmetric_average_call, no_steps, no_sims
    )
    assert_almost_equal(arithmetic, 1.49696, atol=sqrt(Float64(2.0)) * four_standard_errors(4.89, no_sims))
    # on the same paths the arithmetic average is never below the geometric
    assert_true(arithmetic > geometric)


def test_simulated_lookback_options() raises:
    # Example of section 16.5, C++ Code 16.5 with the payoffs of C++ Code 16.7.
    # Book output: analytical lookback put = 16.2665, simulated lookback put
    # = 14.9846, analytical lookback call = 22.8089, simulated lookback call
    # = 21.9336. The simulation observes the price at 250 dates only, which
    # misses part of the extremes, so it lies below the analytical value for
    # continuous observation; the references are the book's simulated values.
    var S: Float64 = 100.0
    var r: Float64 = 0.10
    var time: Float64 = 1.0
    var sigma: Float64 = 0.25
    var no_sims = 10000
    var no_steps = 250
    random_seed(167)
    # put: standard deviation of one discounted payoff 11.26, so
    # 4 * sqrt(2) standard errors are 0.637
    var put = derivative_price_simulate_european_option_generic(
        S, 0.0, r, sigma, time, payoff_lookback_put, no_steps, no_sims
    )
    assert_almost_equal(put, 14.9846, atol=sqrt(Float64(2.0)) * four_standard_errors(11.26, no_sims))
    # the analytical value is 9 standard errors above the expected simulated value
    assert_true(put < option_price_european_lookback_put(S, S, r, 0.0, sigma, time))
    # call: standard deviation of one discounted payoff 19.40, so
    # 4 * sqrt(2) standard errors are 1.097
    var call = derivative_price_simulate_european_option_generic(
        S, 0.0, r, sigma, time, payoff_lookback_call, no_steps, no_sims
    )
    assert_almost_equal(call, 21.9336, atol=sqrt(Float64(2.0)) * four_standard_errors(19.40, no_sims))


def test_simulation_with_control_variate() raises:
    # C++ Code 16.8. Book output: simulated lookback call using control
    # variates = 22.0685. The lookback payoff less the at the money call
    # payoff has standard deviation 5.86 after discounting, so 4 * sqrt(2)
    # standard errors are 0.331.
    var S: Float64 = 100.0
    var r: Float64 = 0.10
    var time: Float64 = 1.0
    var sigma: Float64 = 0.25
    var no_sims = 10000
    var no_steps = 250
    random_seed(168)
    var call = derivative_price_simulate_european_option_generic_with_control_variate(
        S, 0.0, r, sigma, time, payoff_lookback_call, no_steps, no_sims
    )
    assert_almost_equal(call, 22.0685, atol=sqrt(Float64(2.0)) * four_standard_errors(5.86, no_sims))
    # with the control variate the estimate is precise enough to see the
    # effect of discrete observation: the analytical value 22.8089 is 13
    # standard errors above the expected simulated value
    assert_true(call < option_price_european_lookback_call(S, S, r, 0.0, sigma, time))
    # When the payoff is the control variate itself (an at the money call on
    # the terminal price) the simulated payoffs cancel and the Black Scholes
    # price is returned whatever the random numbers.
    var c = derivative_price_simulate_european_option_generic_with_control_variate(
        S, S, r, sigma, time, payoff_terminal_call, 10, 1000
    )
    assert_almost_equal(c, option_price_call_black_scholes(S, S, r, sigma, time), atol=1e-9)
    assert_almost_equal(c, 14.9758, atol=1e-4)
    # and the plain generic routine prices the same payoff with sampling
    # error: standard deviation of one discounted payoff 19.92
    random_seed(169)
    c = derivative_price_simulate_european_option_generic(S, S, r, sigma, time, payoff_terminal_call, 10, 50000)
    assert_almost_equal(c, 14.9758, atol=four_standard_errors(19.92, 50000))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
