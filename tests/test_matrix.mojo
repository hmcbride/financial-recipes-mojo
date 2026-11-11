from std.testing import TestSuite, assert_almost_equal
from finrecipes.matrix import mat_solve, mat_inverse, mat_mul, mat_vec, mat_transpose, mat_identity, quadratic_form


def test_solve_and_inverse() raises:
    # Chapter 2 example: A = [1 2; 4 5], b = [5; 14] -> x = [1; 2]
    var A: List[List[Float64]] = [[1.0, 2.0], [4.0, 5.0]]
    var b: List[Float64] = [5.0, 14.0]
    var x = mat_solve(A, b)
    assert_almost_equal(x[0], 1.0, atol=1e-10)
    assert_almost_equal(x[1], 2.0, atol=1e-10)
    var inv = mat_inverse(A)
    var prod = mat_mul(A, inv)
    var eye = mat_identity(2)
    for i in range(2):
        for j in range(2):
            assert_almost_equal(prod[i][j], eye[i][j], atol=1e-10)


def test_discount_factors_from_bonds() raises:
    # Chapter 5 example: d = C^-1 B
    var C: List[List[Float64]] = [
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 110.0, 0.0, 0.0],
        [8.0, 8.0, 8.0, 8.0, 108.0, 0.0],
        [9.0, 9.0, 9.0, 9.0, 9.0, 109.0],
    ]
    var B: List[Float64] = [96.0, 94.0, 92.0, 118.0, 109.0, 112.0]
    var d = mat_solve(C, B)
    var expected: List[Float64] = [0.9600, 0.9400, 0.9200, 0.8164, 0.7399, 0.6662]
    for i in range(6):
        assert_almost_equal(d[i], expected[i], atol=1e-4)


def test_products() raises:
    var A: List[List[Float64]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    var At = mat_transpose(A)
    assert_almost_equal(At[2][1], 6.0)
    var v = mat_vec(A, [1.0, 1.0, 1.0])
    assert_almost_equal(v[0], 6.0)
    assert_almost_equal(v[1], 15.0)
    var S: List[List[Float64]] = [[2.0, 0.0], [0.0, 3.0]]
    assert_almost_equal(quadratic_form([1.0, 2.0], S, [1.0, 2.0]), 14.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
