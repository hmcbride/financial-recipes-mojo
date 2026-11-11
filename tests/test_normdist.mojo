from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.normdist import n, N, N2
from finrecipes.random import random_seed, random_seed_c_rand, random_uniform_0_1, random_normal


def test_cumulative_normal() raises:
    # Book, Appendix A example: N(0) = 0.5, N(0,0,0) = 0.25
    assert_almost_equal(N(0.0), 0.5, atol=1e-6)
    assert_almost_equal(N2(0.0, 0.0, 0.0), 0.25, atol=1e-6)
    assert_almost_equal(N(1.96), 0.975, atol=1e-4)
    assert_almost_equal(N(-1.0) + N(1.0), 1.0, atol=1e-7)
    assert_almost_equal(N(7.0), 1.0)
    assert_almost_equal(N(-7.0), 0.0)


def test_density() raises:
    assert_almost_equal(n(0.0), 0.3989423, atol=1e-6)
    assert_almost_equal(n(1.0), n(-1.0), atol=1e-12)


def test_bivariate_normal() raises:
    # independent variables: joint probability is the product
    assert_almost_equal(N2(0.5, -0.3, 0.0), N(0.5) * N(-0.3), atol=1e-4)
    # perfectly correlated limit approaches min(N(a), N(b))
    assert_almost_equal(N2(0.2, 1.0, 0.999), N(0.2), atol=1e-2)
    assert_almost_equal(N2(1.0, 1.0, 0.5), N2(1.0, 1.0, 0.5), atol=1e-12)
    assert_almost_equal(N2(0.0, 0.0, 0.5), 1.0 / 3.0, atol=1e-4)


def test_random_numbers() raises:
    random_seed(42)
    var sum: Float64 = 0.0
    var sum2: Float64 = 0.0
    var usum: Float64 = 0.0
    var count = 20000
    for _ in range(count):
        var u = random_uniform_0_1()
        assert_true(u >= 0.0 and u < 1.0)
        usum += u
        var z = random_normal()
        sum += z
        sum2 += z * z
    assert_almost_equal(usum / Float64(count), 0.5, atol=0.02)
    assert_almost_equal(sum / Float64(count), 0.0, atol=0.03)
    assert_almost_equal(sum2 / Float64(count), 1.0, atol=0.05)


def test_c_library_generator_reproduces_the_book() raises:
    # Book, Appendix A example: the first numbers a C program gets from
    # rand(), and the normal numbers that follow them.
    random_seed_c_rand(1)
    var u: List[Float64] = [0.840188, 0.394383, 0.783099, 0.79844, 0.911647]
    for i in range(5):
        assert_almost_equal(random_uniform_0_1(), u[i], atol=1e-6)
    var z: List[Float64] = [-1.07224, 0.925946, 2.70202, 1.36918, 0.0187313]
    for i in range(5):
        assert_almost_equal(random_normal(), z[i], atol=1e-5)
    # reseeding restarts the sequence
    random_seed_c_rand(1)
    assert_almost_equal(random_uniform_0_1(), 0.840188, atol=1e-6)
    # random_seed switches back to Mojo's generator
    random_seed(7)
    var a = random_uniform_0_1()
    random_seed(7)
    assert_almost_equal(random_uniform_0_1(), a, atol=1e-15)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
