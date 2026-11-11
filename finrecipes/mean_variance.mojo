"""Chapter 6: the mean variance frontier (Matlab Code 6.1 - 6.4, C++ Code 6.1 - 6.4).

The book does the matrix work with Matlab, Newmat and IT++. The Newmat and
IT++ listings (C++ Code 6.1 / 6.2 and C++ Code 6.3 / 6.4) are the same
algorithm written against two libraries, so each pair is ported once, on
top of `finrecipes/matrix.mojo`. Vectors are `List[Float64]`, matrices are
`List[List[Float64]]` stored row by row.
"""

from std.math import sqrt
from .matrix import mat_inverse, mat_vec, vec_dot, vec_zeros, quadratic_form


def mv_calculate_mean(e: List[Float64], w: List[Float64]) -> Float64:
    """Expected portfolio return e'w (C++ Code 6.1 and 6.2).

    e: expected asset returns, w: portfolio weights.
    """
    return vec_dot(e, w)


def mv_calculate_variance(V: List[List[Float64]], w: List[Float64]) -> Float64:
    """Portfolio variance w'Vw (C++ Code 6.1 and 6.2).

    V: covariance matrix of asset returns, w: portfolio weights.
    """
    return quadratic_form(w, V, w)


def mv_calculate_st_dev(V: List[List[Float64]], w: List[Float64]) -> Float64:
    """Portfolio standard deviation sqrt(w'Vw) (C++ Code 6.1 and 6.2)."""
    var variance = mv_calculate_variance(V, w)
    return sqrt(variance)


def mv_calculate_portfolio_given_mean_unconstrained(
    e: List[Float64], V: List[List[Float64]], r: Float64
) raises -> List[Float64]:
    """Unconstrained frontier portfolio for expected return r (C++ Code 6.3 and 6.4).

    e: expected asset returns, V: covariance matrix (must be invertible),
    r: required portfolio return. Returns the weights w = g + h r.
    """
    var no_assets = len(e)
    var ones = List[Float64](length=no_assets, fill=1.0)
    var Vinv = mat_inverse(V)  # inverse of V
    var Vinv1 = mat_vec(Vinv, ones)
    var Vinve = mat_vec(Vinv, e)
    var a = vec_dot(ones, Vinve)
    var b = vec_dot(e, Vinve)
    var c = vec_dot(ones, Vinv1)
    var d = b * c - a * a
    var w = vec_zeros(no_assets)
    for i in range(no_assets):
        var g = (Vinv1[i] * b - Vinve[i] * a) * (1.0 / d)
        var h = (Vinve[i] * c - Vinv1[i] * a) * (1.0 / d)
        w[i] = g + h * r
    return w^


struct MinVarianceFrontier(Copyable, Movable):
    """The quantities that describe the minimum variance frontier (section 6.3).

    Every frontier portfolio has weights w = g + h E[r_p]. These are the
    variables printed by the Matlab example that precedes Matlab Code 6.1.
    """

    var a: Float64  # 1' V^-1 e
    var b: Float64  # e' V^-1 e
    var c: Float64  # 1' V^-1 1
    var d: Float64  # b c - a^2, the determinant of [b a; a c]
    var g: List[Float64]  # weights of the frontier portfolio with mean 0
    var h: List[Float64]  # g + h is the frontier portfolio with mean 1

    def __init__(
        out self, a: Float64, b: Float64, c: Float64, d: Float64, g: List[Float64], h: List[Float64]
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.g = g.copy()
        self.h = h.copy()

    def weights(self, r: Float64) -> List[Float64]:
        """Weights w = g + h r of the frontier portfolio with expected return r."""
        var w = vec_zeros(len(self.g))
        for i in range(len(self.g)):
            w[i] = self.g[i] + self.h[i] * r
        return w^

    def global_minimum_variance_mean(self) -> Float64:
        """Expected return a/c of the global minimum variance portfolio (section 6.4)."""
        return self.a / self.c

    def global_minimum_variance_variance(self) -> Float64:
        """Variance 1/c of the global minimum variance portfolio (section 6.4)."""
        return 1.0 / self.c


def min_variance_frontier(e: List[Float64], V: List[List[Float64]]) raises -> MinVarianceFrontier:
    """The frontier quantities a, b, c, d, g and h computed in Matlab Code 6.1.

    Follows the Matlab statements line by line:
        a = ones(1,n)*inv(V)*e          b = e'*inv(V)*e
        c = ones(1,n)*inv(V)*ones(n,1)  d = det([b a; a c])
        g = 1/d*(b*ones(1,n) - a*e')*inv(V)
        h = 1/d*(c*e' - a*ones(1,n))*inv(V)
    """
    var n = len(e)
    var Vinv = mat_inverse(V)
    var ones = List[Float64](length=n, fill=1.0)
    var a = quadratic_form(ones, Vinv, e)
    var b = quadratic_form(e, Vinv, e)
    var c = quadratic_form(ones, Vinv, ones)
    var d = b * c - a * a  # determinant of the 2x2 matrix A = [b a; a c]
    var g = vec_zeros(n)
    var h = vec_zeros(n)
    for j in range(n):  # row vector times inv(V): column j of the product
        var gj: Float64 = 0.0
        var hj: Float64 = 0.0
        for i in range(n):
            gj += (b * ones[i] - a * e[i]) * Vinv[i][j]
            hj += (c * e[i] - a * ones[i]) * Vinv[i][j]
        g[j] = gj / d
        h[j] = hj / d
    return MinVarianceFrontier(a, b, c, d, g, h)


def min_variance_portfolio(e: List[Float64], V: List[List[Float64]], r: Float64) raises -> List[Float64]:
    """Minimum variance portfolio for a given return r (Matlab Code 6.1)."""
    var frontier = min_variance_frontier(e, V)
    return frontier.weights(r)


def mean(x: List[Float64]) -> Float64:
    """Arithmetic mean (Matlab `mean`)."""
    var s: Float64 = 0.0
    for i in range(len(x)):
        s += x[i]
    return s / Float64(len(x))


def cov(x: List[Float64], y: List[Float64]) -> Float64:
    """Sample covariance of two series, normalised by n - 1 (Matlab `cov`)."""
    var mx = mean(x)
    var my = mean(y)
    var s: Float64 = 0.0
    for i in range(len(x)):
        s += (x[i] - mx) * (y[i] - my)
    return s / Float64(len(x) - 1)


def variance(x: List[Float64]) -> Float64:
    """Sample variance, normalised by n - 1 (Matlab `var`)."""
    return cov(x, x)


def stdev(x: List[Float64]) -> Float64:
    """Sample standard deviation, normalised by n - 1 (Matlab `std`).

    Named `stdev` because `std` is the name of the Mojo standard library package.
    """
    return sqrt(variance(x))


def _excess(r: List[Float64], rf: List[Float64]) -> List[Float64]:
    """The series r - rf."""
    var x = vec_zeros(len(r))
    for i in range(len(r)):
        x[i] = r[i] - rf[i]
    return x^


def sharpe(r: List[Float64], rf: List[Float64]) -> Float64:
    """Sharpe ratio (Matlab Code 6.2).

    r: series of portfolio returns, rf: series of risk free returns over the
    same periods. s = (mean(r) - mean(rf)) / std(r - rf).
    """
    return (mean(r) - mean(rf)) / stdev(_excess(r, rf))


def sharpe(r: List[Float64], rf: Float64) -> Float64:
    """Sharpe ratio with a constant risk free rate (Matlab Code 6.2 with scalar rf)."""
    return (mean(r) - rf) / stdev(r)


def treynor(r: List[Float64], rm: List[Float64], rf: List[Float64]) -> Float64:
    """Treynor ratio (Matlab Code 6.3).

    r: series of portfolio returns, rm: series of market returns, rf: series
    of risk free returns. beta = cov(r, rm) / var(rm), t = mean(r - rf) / beta.
    """
    var beta = cov(r, rm) / variance(rm)
    return mean(_excess(r, rf)) / beta


def treynor(r: List[Float64], rm: List[Float64], rf: Float64) -> Float64:
    """Treynor ratio with a constant risk free rate (Matlab Code 6.3 with scalar rf)."""
    var beta = cov(r, rm) / variance(rm)
    return (mean(r) - rf) / beta


def jensen(r: List[Float64], rm: List[Float64], rf: Float64) -> Float64:
    """Jensen's alpha (Matlab Code 6.4).

    r: series of portfolio returns, rm: series of market returns, rf: the
    risk free rate. alpha = mean(r) - (rf + beta (mean(rm) - rf)).
    """
    var beta = cov(r, rm) / variance(rm)
    return mean(r) - (rf + beta * (mean(rm) - rf))
