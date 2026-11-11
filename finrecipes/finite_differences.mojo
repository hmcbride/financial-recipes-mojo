"""Chapter 13: finite differences (C++ Code 13.1 - 13.5).

The price grid runs from 0 to 2S in M steps, so the current price sits on
grid point M/2; M is rounded up to an even number.

The book lists the put routines. The call routines here are their mirror
images (payoff and boundary values swapped) and are not book listings.

The implicit routines need the solution of a linear system A F = B at every
time step. The book does this with the Newmat (C++ Code 13.3, 13.5) and IT++
(C++ Code 13.4) matrix libraries, multiplying with the inverse of A. Here A
is built with `mat_zeros` from `matrix.mojo` and, because it is tridiagonal,
solved by `mat_solve_tridiagonal`: the Gaussian elimination of `mat_solve`
restricted to the band, with the same operations in the same order and so
the same result, in a time proportional to the size of the grid. That is
also an answer to Exercise 13.1.
"""

from .matrix import mat_zeros


def mat_solve_tridiagonal(a: List[List[Float64]], b: List[Float64]) raises -> List[Float64]:
    """Solve A x = b for a tridiagonal matrix A stored as a full matrix.

    Gaussian elimination with partial pivoting, as in `mat_solve`, but only
    the three diagonals of A are read and each column is eliminated from the
    single row below it. A row exchange can add one more diagonal above the
    band, which is carried in `up2`.
    """
    var n = len(b)
    var low = List[Float64](length=n, fill=0.0)  # A[i][i-1]
    var diag = List[Float64](length=n, fill=0.0)  # A[i][i]
    var up1 = List[Float64](length=n, fill=0.0)  # A[i][i+1]
    var up2 = List[Float64](length=n, fill=0.0)  # A[i][i+2], filled by row exchanges
    for i in range(n):
        diag[i] = a[i][i]
        if i > 0:
            low[i] = a[i][i - 1]
        if i + 1 < n:
            up1[i] = a[i][i + 1]
    var x = b.copy()
    for col in range(n):
        var below = col + 1
        if below < n and abs(low[below]) > abs(diag[col]):
            # exchange rows col and col+1
            var tmp = diag[col]
            diag[col] = low[below]
            low[below] = tmp
            tmp = up1[col]
            up1[col] = diag[below]
            diag[below] = tmp
            tmp = up2[col]
            up2[col] = up1[below]
            up1[below] = tmp
            tmp = x[col]
            x[col] = x[below]
            x[below] = tmp
        if abs(diag[col]) < 1e-14:
            raise Error("mat_solve_tridiagonal: matrix is singular")
        if below < n:
            var factor = low[below] / diag[col]
            if factor != 0.0:
                diag[below] -= factor * up1[col]
                up1[below] -= factor * up2[col]
                x[below] -= factor * x[col]
    for i in range(n - 1, -1, -1):
        var s = x[i]
        if i + 1 < n:
            s -= up1[i] * x[i + 1]
        if i + 2 < n:
            s -= up2[i] * x[i + 2]
        x[i] = s / diag[i]
    return x^


def option_price_put_european_finite_diff_explicit(
    S: Float64, X: Float64, r: Float64, sigma: Float64, time: Float64, no_S_steps: Int, no_t_steps: Int
) -> Float64:
    """European put by explicit finite differences (C++ Code 13.1).

    The explicit scheme may fail to converge for some combinations of inputs.
    """
    var sigma_sqr = sigma * sigma
    var M = no_S_steps
    if (no_S_steps % 2) == 1:
        M += 1  # need no_S_steps to be even:
    var delta_S = 2.0 * S / Float64(M)
    var S_values = List[Float64](length=M + 1, fill=0.0)
    for m in range(M + 1):
        S_values[m] = Float64(m) * delta_S
    var N = no_t_steps
    var delta_t = time / Float64(N)
    var a = List[Float64](length=M, fill=0.0)
    var b = List[Float64](length=M, fill=0.0)
    var c = List[Float64](length=M, fill=0.0)
    var r1 = 1.0 / (1.0 + r * delta_t)
    var r2 = delta_t / (1.0 + r * delta_t)
    for j in range(1, M):
        var fj = Float64(j)
        a[j] = r2 * 0.5 * fj * (-r + sigma_sqr * fj)
        b[j] = r1 * (1.0 - sigma_sqr * fj * fj * delta_t)
        c[j] = r2 * 0.5 * fj * (r + sigma_sqr * fj)
    var f_next = List[Float64](length=M + 1, fill=0.0)
    for m in range(M + 1):
        f_next[m] = max(0.0, X - S_values[m])
    var f = List[Float64](length=M + 1, fill=0.0)
    for _ in range(N - 1, -1, -1):
        f[0] = X
        for m in range(1, M):
            f[m] = a[m] * f_next[m - 1] + b[m] * f_next[m] + c[m] * f_next[m + 1]
        f[M] = 0.0
        for m in range(M + 1):
            f_next[m] = f[m]
    return f[M // 2]


def option_price_put_american_finite_diff_explicit(
    S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64, no_S_steps: Int, no_t_steps: Int
) -> Float64:
    """American put by explicit finite differences (C++ Code 13.2, Matlab Code 13.1)."""
    var sigma_sqr = sigma * sigma
    var M = no_S_steps + (no_S_steps % 2)  # need no_S_steps to be even:
    var delta_S = 2.0 * S / Float64(M)
    var S_values = List[Float64](length=M + 1, fill=0.0)
    for m in range(M + 1):
        S_values[m] = Float64(m) * delta_S
    var N = no_t_steps
    var delta_t = time / Float64(N)
    var a = List[Float64](length=M, fill=0.0)
    var b = List[Float64](length=M, fill=0.0)
    var c = List[Float64](length=M, fill=0.0)
    var r1 = 1.0 / (1.0 + r * delta_t)
    var r2 = delta_t / (1.0 + r * delta_t)
    for j in range(1, M):
        var fj = Float64(j)
        a[j] = r2 * 0.5 * fj * (-r + sigma_sqr * fj)
        b[j] = r1 * (1.0 - sigma_sqr * fj * fj * delta_t)
        c[j] = r2 * 0.5 * fj * (r + sigma_sqr * fj)
    var f_next = List[Float64](length=M + 1, fill=0.0)
    for m in range(M + 1):
        f_next[m] = max(0.0, K - S_values[m])
    var f = List[Float64](length=M + 1, fill=0.0)
    for _ in range(N - 1, -1, -1):
        f[0] = K
        for m in range(1, M):
            f[m] = a[m] * f_next[m - 1] + b[m] * f_next[m] + c[m] * f_next[m + 1]
            f[m] = max(f[m], K - S_values[m])  # check for exercise
        f[M] = 0.0
        for m in range(M + 1):
            f_next[m] = f[m]
    return f[M // 2]


def option_price_call_european_finite_diff_explicit(
    S: Float64, X: Float64, r: Float64, sigma: Float64, time: Float64, no_S_steps: Int, no_t_steps: Int
) -> Float64:
    """European call by explicit finite differences (call version of C++ Code 13.1).

    Not listed in the book. The option is worthless at a zero price and worth
    its exercise value at the upper edge of the grid.
    """
    var sigma_sqr = sigma * sigma
    var M = no_S_steps + (no_S_steps % 2)  # need no_S_steps to be even:
    var delta_S = 2.0 * S / Float64(M)
    var S_values = List[Float64](length=M + 1, fill=0.0)
    for m in range(M + 1):
        S_values[m] = Float64(m) * delta_S
    var N = no_t_steps
    var delta_t = time / Float64(N)
    var a = List[Float64](length=M, fill=0.0)
    var b = List[Float64](length=M, fill=0.0)
    var c = List[Float64](length=M, fill=0.0)
    var r1 = 1.0 / (1.0 + r * delta_t)
    var r2 = delta_t / (1.0 + r * delta_t)
    for j in range(1, M):
        var fj = Float64(j)
        a[j] = r2 * 0.5 * fj * (-r + sigma_sqr * fj)
        b[j] = r1 * (1.0 - sigma_sqr * fj * fj * delta_t)
        c[j] = r2 * 0.5 * fj * (r + sigma_sqr * fj)
    var f_next = List[Float64](length=M + 1, fill=0.0)
    for m in range(M + 1):
        f_next[m] = max(0.0, S_values[m] - X)
    var f = List[Float64](length=M + 1, fill=0.0)
    for _ in range(N - 1, -1, -1):
        f[0] = 0.0
        for m in range(1, M):
            f[m] = a[m] * f_next[m - 1] + b[m] * f_next[m] + c[m] * f_next[m + 1]
        f[M] = S_values[M] - X
        for m in range(M + 1):
            f_next[m] = f[m]
    return f[M // 2]


def option_price_call_american_finite_diff_explicit(
    S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64, no_S_steps: Int, no_t_steps: Int
) -> Float64:
    """American call by explicit finite differences (call version of C++ Code 13.2).

    Not listed in the book.
    """
    var sigma_sqr = sigma * sigma
    var M = no_S_steps + (no_S_steps % 2)  # need no_S_steps to be even:
    var delta_S = 2.0 * S / Float64(M)
    var S_values = List[Float64](length=M + 1, fill=0.0)
    for m in range(M + 1):
        S_values[m] = Float64(m) * delta_S
    var N = no_t_steps
    var delta_t = time / Float64(N)
    var a = List[Float64](length=M, fill=0.0)
    var b = List[Float64](length=M, fill=0.0)
    var c = List[Float64](length=M, fill=0.0)
    var r1 = 1.0 / (1.0 + r * delta_t)
    var r2 = delta_t / (1.0 + r * delta_t)
    for j in range(1, M):
        var fj = Float64(j)
        a[j] = r2 * 0.5 * fj * (-r + sigma_sqr * fj)
        b[j] = r1 * (1.0 - sigma_sqr * fj * fj * delta_t)
        c[j] = r2 * 0.5 * fj * (r + sigma_sqr * fj)
    var f_next = List[Float64](length=M + 1, fill=0.0)
    for m in range(M + 1):
        f_next[m] = max(0.0, S_values[m] - K)
    var f = List[Float64](length=M + 1, fill=0.0)
    for _ in range(N - 1, -1, -1):
        f[0] = 0.0
        for m in range(1, M):
            f[m] = a[m] * f_next[m - 1] + b[m] * f_next[m] + c[m] * f_next[m + 1]
            f[m] = max(f[m], S_values[m] - K)  # check for exercise
        f[M] = S_values[M] - K
        for m in range(M + 1):
            f_next[m] = f[m]
    return f[M // 2]


def option_price_put_american_finite_diff_implicit(
    S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64, no_S_steps: Int, no_t_steps: Int
) raises -> Float64:
    """American put by implicit finite differences (C++ Code 13.3 and 13.4, Matlab Code 13.2).

    The Newmat and IT++ listings are the same algorithm with two matrix
    libraries, so there is one port.
    """
    var sigma_sqr = sigma * sigma
    var M = no_S_steps + (no_S_steps % 2)  # need no_S_steps to be even:
    var delta_S = 2.0 * S / Float64(M)
    var S_values = List[Float64](length=M + 1, fill=0.0)
    for m in range(M + 1):
        S_values[m] = Float64(m) * delta_S
    var N = no_t_steps
    var delta_t = time / Float64(N)
    var A = mat_zeros(M + 1, M + 1)
    A[0][0] = 1.0
    for j in range(1, M):
        var fj = Float64(j)
        A[j][j - 1] = 0.5 * fj * delta_t * (r - sigma_sqr * fj)  # a[j]
        A[j][j] = 1.0 + delta_t * (r + sigma_sqr * fj * fj)  # b[j]
        A[j][j + 1] = 0.5 * fj * delta_t * (-r - sigma_sqr * fj)  # c[j]
    A[M][M] = 1.0
    var B = List[Float64](length=M + 1, fill=0.0)
    for m in range(M + 1):
        B[m] = max(0.0, K - S_values[m])
    var F = mat_solve_tridiagonal(A, B)
    for _ in range(N - 1, 0, -1):
        B = F.copy()
        F = mat_solve_tridiagonal(A, B)
        for m in range(1, M):  # now check for exercise
            F[m] = max(F[m], K - S_values[m])
    return F[M // 2]


def option_price_put_european_finite_diff_implicit(
    S: Float64, K: Float64, r: Float64, sigma: Float64, time: Float64, no_S_steps: Int, no_t_steps: Int
) raises -> Float64:
    """European put by implicit finite differences (C++ Code 13.5)."""
    var sigma_sqr = sigma * sigma
    var M = no_S_steps + (no_S_steps % 2)  # need no_S_steps to be even:
    var delta_S = 2.0 * S / Float64(M)
    var S_values = List[Float64](length=M + 1, fill=0.0)
    for m in range(M + 1):
        S_values[m] = Float64(m) * delta_S
    var N = no_t_steps
    var delta_t = time / Float64(N)
    var A = mat_zeros(M + 1, M + 1)
    A[0][0] = 1.0
    for j in range(1, M):
        var fj = Float64(j)
        A[j][j - 1] = 0.5 * fj * delta_t * (r - sigma_sqr * fj)  # a[j]
        A[j][j] = 1.0 + delta_t * (r + sigma_sqr * fj * fj)  # b[j]
        A[j][j + 1] = 0.5 * fj * delta_t * (-r - sigma_sqr * fj)  # c[j]
    A[M][M] = 1.0
    var B = List[Float64](length=M + 1, fill=0.0)
    for m in range(M + 1):
        B[m] = max(0.0, K - S_values[m])
    var F = mat_solve_tridiagonal(A, B)
    for _ in range(N - 1, 0, -1):
        B = F.copy()
        F = mat_solve_tridiagonal(A, B)
    return F[M // 2]
