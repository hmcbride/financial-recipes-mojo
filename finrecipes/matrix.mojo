"""Small dense linear algebra helpers.

The book relies on external C++ matrix libraries (Newmat, IT++) and Matlab
for its matrix work. This module supplies the few operations those listings
need. A matrix is a `List[List[Float64]]` stored row by row, a vector is a
`List[Float64]`.
"""


def vec_zeros(n: Int) -> List[Float64]:
    return List[Float64](length=n, fill=0.0)


def mat_zeros(rows: Int, cols: Int) -> List[List[Float64]]:
    var m = List[List[Float64]]()
    for _ in range(rows):
        m.append(List[Float64](length=cols, fill=0.0))
    return m^


def mat_identity(n: Int) -> List[List[Float64]]:
    var m = mat_zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m^


def mat_copy(a: List[List[Float64]]) -> List[List[Float64]]:
    var m = List[List[Float64]]()
    for i in range(len(a)):
        m.append(a[i].copy())
    return m^


def mat_transpose(a: List[List[Float64]]) -> List[List[Float64]]:
    var rows = len(a)
    var cols = len(a[0])
    var m = mat_zeros(cols, rows)
    for i in range(rows):
        for j in range(cols):
            m[j][i] = a[i][j]
    return m^


def mat_mul(a: List[List[Float64]], b: List[List[Float64]]) -> List[List[Float64]]:
    var rows = len(a)
    var inner = len(b)
    var cols = len(b[0])
    var m = mat_zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            var s: Float64 = 0.0
            for k in range(inner):
                s += a[i][k] * b[k][j]
            m[i][j] = s
    return m^


def mat_vec(a: List[List[Float64]], x: List[Float64]) -> List[Float64]:
    """Matrix times column vector."""
    var out = vec_zeros(len(a))
    for i in range(len(a)):
        var s: Float64 = 0.0
        for j in range(len(x)):
            s += a[i][j] * x[j]
        out[i] = s
    return out^


def vec_dot(x: List[Float64], y: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(x)):
        s += x[i] * y[i]
    return s


def quadratic_form(x: List[Float64], a: List[List[Float64]], y: List[Float64]) -> Float64:
    """x' A y."""
    return vec_dot(x, mat_vec(a, y))


def mat_solve(a: List[List[Float64]], b: List[Float64]) raises -> List[Float64]:
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    var n = len(b)
    var m = mat_copy(a)
    var x = b.copy()
    for col in range(n):
        var pivot = col
        var best = abs(m[col][col])
        for row in range(col + 1, n):
            if abs(m[row][col]) > best:
                best = abs(m[row][col])
                pivot = row
        if best < 1e-14:
            raise Error("mat_solve: matrix is singular")
        if pivot != col:
            for k in range(n):
                var tmp = m[col][k]
                m[col][k] = m[pivot][k]
                m[pivot][k] = tmp
            var tb = x[col]
            x[col] = x[pivot]
            x[pivot] = tb
        for row in range(col + 1, n):
            var factor = m[row][col] / m[col][col]
            if factor != 0.0:
                for k in range(col, n):
                    m[row][k] -= factor * m[col][k]
                x[row] -= factor * x[col]
    for i in range(n - 1, -1, -1):
        var s = x[i]
        for k in range(i + 1, n):
            s -= m[i][k] * x[k]
        x[i] = s / m[i][i]
    return x^


def mat_inverse(a: List[List[Float64]]) raises -> List[List[Float64]]:
    """Matrix inverse, computed column by column with `mat_solve`."""
    var n = len(a)
    var inv = mat_zeros(n, n)
    for j in range(n):
        var e = vec_zeros(n)
        e[j] = 1.0
        var col = mat_solve(a, e)
        for i in range(n):
            inv[i][j] = col[i]
    return inv^
