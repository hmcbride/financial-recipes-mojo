# Porting notes: C++ to Mojo

These are the conventions used to port the book's C++ listings, written
against Mojo 1.3 (nightly, November 2026). Mojo changes quickly; everything
below was checked by compiling it.

## Build and test

```sh
pixi run mojo run -I . tests/test_black_scholes.mojo   # one test file
pixi run test                                          # all test files
```

`-I .` puts the project root on the import path, so tests can
`from finrecipes.<module> import ...`. Modules inside the package import
each other with relative imports: `from .normdist import N`.

## Language rules that differ from older Mojo

- `fn` is gone. Every function is `def`. Arguments are typed; a function
  that can raise is declared `def f(...) raises -> T:`.
- The standard library is under `std.`: `from std.math import exp, log,
  sqrt, pow, cos, sin, atan`, `from std.testing import ...`,
  `from std.random import ...`.
- `abs`, `min`, `max`, `len`, `range`, `print` are builtins.
- Declare variables with `var`. Write float literals with a decimal point
  and annotate when the type would otherwise be a literal type:
  `var x: Float64 = 0.0`. Compile-time constants: `comptime PI: Float64 = 3.14`.
- Conversions: `Float64(i)` for Int to float, `Int(x)` for a `Float64`
  variable (truncates). `Int(3.7)` on a literal does not compile.
- `pow(x, y)` and `x ** y` work on `Float64`. For integer powers of small
  degree prefer `x * x`. `std.math` `exp`/`pow` are accurate to roughly
  1e-13 relative, not to the last bit.
- Lists: `List[Float64]()`, `List[Float64](length=n, fill=0.0)`,
  `var v: List[Float64] = [1.0, 2.0]`, `v.append(x)`, `v.copy()`.
  `[0.0] * n` does not compile.
- Lists are not implicitly copyable. Return a local list with a transfer:
  `return v^`. Store an argument list into a field with `.copy()`.
  A read-only list argument is just `v: List[Float64]`; one the function
  modifies is `mut v: List[Float64]`.
- Structs: `struct Name(Copyable, Movable):`, fields declared with `var`,
  constructor `def __init__(out self, ...)`, mutating methods take
  `mut self`.
- Traits replace C++ base classes, and can have default method bodies.
  Functions generic over a trait: `def f[T: TermStructure](ts: T, ...)`.
  A generic struct refers to its parameter as `Self.T`, and a field of
  type `Self.T` needs the bound `T: SomeTrait & Copyable & Deinitable`.
- Function-pointer arguments (the book's "function prototypes"):
  `payoff: def(Float64, Float64) thin -> Float64`.
- Several return values: return a small struct or a `Tuple[Float64, Float64]`
  (`return (a, b)`, read with `t[0]`). C++ reference out-parameters become a
  returned struct, for example `Greeks` in `black_scholes.mojo`.
- Reverse loops: `for i in range(n - 1, -1, -1):`.
- A `var` that is assigned and never read produces a warning; keep the
  build warning-free. `var x: Float64` with no initialiser, assigned in
  both branches of an `if`, compiles and avoids that warning.
- A small value struct that is returned by value and then copied around
  freely (such as `Date`) needs `ImplicitlyCopyable` rather than `Copyable`.
- Mojo has no `++` / `--`; the date class exposes them as
  `pre_increment`, `post_increment`, `pre_decrement`, `post_decrement`.

## Tests

Each test file is a program whose `main` discovers the `test_*` functions:

```mojo
from std.testing import TestSuite, assert_almost_equal, assert_true

def test_something() raises:
    assert_almost_equal(value, 5.45325, atol=1e-5)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

Expected values come from the "Output from C++ program" boxes in the book,
with a tolerance matching the printed digits.

## Naming

- Routine names keep the book's names in snake_case
  (`option_price_call_black_scholes`).
- `N(z)` is the cumulative normal, `n(z)` the density, `N2(a, b, rho)` the
  bivariate cumulative normal (overloaded `N` in the book).
- C++ classes become CamelCase structs (`term_structure_class_flat` is
  `TermStructureFlat`).
