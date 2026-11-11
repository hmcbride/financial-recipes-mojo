"""The book's Monte Carlo examples, reproduced digit for digit.

The book's programs draw from the C library rand(), started afresh for each
example. `random_seed_c_rand(1)` gives the same sequence, so the simulated
values printed in the book can be asserted exactly.
"""

from std.testing import TestSuite, assert_almost_equal
from finrecipes.random import random_seed, random_seed_c_rand
from finrecipes.simulation import (
    option_price_call_european_simulated,
    option_price_put_european_simulated,
    option_price_delta_call_european_simulated,
    option_price_delta_put_european_simulated,
    derivative_price_simulate_european_option_generic,
    derivative_price_simulate_european_option_generic_with_control_variate,
    derivative_price_simulate_european_option_generic_with_antithetic_variate,
    payoff_call,
    payoff_put,
    payoff_cash_or_nothing_call,
    payoff_asset_or_nothing_call,
)
from finrecipes.exotics import (
    payoff_arithmetric_average_call,
    payoff_geometric_average_call,
    payoff_lookback_call,
    payoff_lookback_put,
)
from finrecipes.exotics import derivative_price_simulate_european_option_generic as path_generic
from finrecipes.exotics import (
    derivative_price_simulate_european_option_generic_with_control_variate as path_generic_cv,
)


def test_chapter_14_simulated_prices() raises:
    random_seed_c_rand(1)
    assert_almost_equal(option_price_call_european_simulated(100.0, 100.0, 0.1, 0.25, 1.0, 5000), 14.8404, atol=1e-4)
    assert_almost_equal(option_price_put_european_simulated(100.0, 100.0, 0.1, 0.25, 1.0, 5000), 5.74588, atol=1e-5)


def test_chapter_14_simulated_deltas() raises:
    random_seed_c_rand(1)
    assert_almost_equal(
        option_price_delta_call_european_simulated(100.0, 100.0, 0.1, 0.25, 1.0, 5000), 0.701484, atol=1e-6
    )
    assert_almost_equal(
        option_price_delta_put_european_simulated(100.0, 100.0, 0.1, 0.25, 1.0, 5000), -0.307211, atol=1e-6
    )


def test_chapter_14_control_and_antithetic_variates() raises:
    random_seed_c_rand(1)
    assert_almost_equal(
        derivative_price_simulate_european_option_generic(100.0, 100.0, 0.1, 0.25, 1.0, payoff_call, 50000),
        14.995,
        atol=1e-3,
    )
    assert_almost_equal(
        derivative_price_simulate_european_option_generic_with_control_variate(
            100.0, 100.0, 0.1, 0.25, 1.0, payoff_call, 50000
        ),
        14.9758,
        atol=1e-4,
    )
    assert_almost_equal(
        derivative_price_simulate_european_option_generic_with_antithetic_variate(
            100.0, 100.0, 0.1, 0.25, 1.0, payoff_call, 50000
        ),
        14.9919,
        atol=1e-4,
    )
    assert_almost_equal(
        derivative_price_simulate_european_option_generic(100.0, 100.0, 0.1, 0.25, 1.0, payoff_put, 50000),
        5.41861,
        atol=1e-5,
    )
    assert_almost_equal(
        derivative_price_simulate_european_option_generic_with_control_variate(
            100.0, 100.0, 0.1, 0.25, 1.0, payoff_put, 50000
        ),
        5.42541,
        atol=1e-5,
    )
    assert_almost_equal(
        derivative_price_simulate_european_option_generic_with_antithetic_variate(
            100.0, 100.0, 0.1, 0.25, 1.0, payoff_put, 50000
        ),
        5.46043,
        atol=1e-5,
    )


def test_chapter_14_binary_options() raises:
    random_seed_c_rand(1)
    assert_almost_equal(
        derivative_price_simulate_european_option_generic(
            100.0, 100.0, 0.1, 0.25, 1.0, payoff_cash_or_nothing_call, 5000
        ),
        0.547427,
        atol=1e-6,
    )
    assert_almost_equal(
        derivative_price_simulate_european_option_generic_with_control_variate(
            100.0, 100.0, 0.1, 0.25, 1.0, payoff_cash_or_nothing_call, 5000
        ),
        1.02552,
        atol=1e-5,
    )
    assert_almost_equal(
        derivative_price_simulate_european_option_generic_with_antithetic_variate(
            100.0, 100.0, 0.1, 0.25, 1.0, payoff_cash_or_nothing_call, 5000
        ),
        0.549598,
        atol=1e-6,
    )
    assert_almost_equal(
        derivative_price_simulate_european_option_generic(
            100.0, 100.0, 0.1, 0.25, 1.0, payoff_asset_or_nothing_call, 5000
        ),
        70.5292,
        atol=1e-4,
    )
    assert_almost_equal(
        derivative_price_simulate_european_option_generic_with_control_variate(
            100.0, 100.0, 0.1, 0.25, 1.0, payoff_asset_or_nothing_call, 5000
        ),
        69.8451,
        atol=1e-4,
    )
    assert_almost_equal(
        derivative_price_simulate_european_option_generic_with_antithetic_variate(
            100.0, 100.0, 0.1, 0.25, 1.0, payoff_asset_or_nothing_call, 5000
        ),
        70.2205,
        atol=1e-4,
    )


def test_chapter_16_path_dependent_options() raises:
    random_seed_c_rand(1)
    assert_almost_equal(
        path_generic(100.0, 120.0, 0.10, 0.25, 1.0, payoff_arithmetric_average_call, 250, 10000), 1.49696, atol=1e-5
    )
    assert_almost_equal(
        path_generic(100.0, 120.0, 0.10, 0.25, 1.0, payoff_geometric_average_call, 250, 10000), 1.38017, atol=1e-5
    )
    assert_almost_equal(path_generic(100.0, 0.0, 0.10, 0.25, 1.0, payoff_lookback_put, 250, 10000), 14.9846, atol=1e-4)
    assert_almost_equal(path_generic(100.0, 0.0, 0.10, 0.25, 1.0, payoff_lookback_call, 250, 10000), 21.9336, atol=1e-4)
    assert_almost_equal(
        path_generic_cv(100.0, 0.0, 0.10, 0.25, 1.0, payoff_lookback_call, 250, 10000), 22.0685, atol=1e-4
    )
    random_seed(1)  # leave Mojo's generator active


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
