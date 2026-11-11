# financial-recipes-mojo

A Mojo port of the programs in Bernt Arne Ødegaard's
*Financial Numerical Recipes in C++* (June 2014 edition).

Every C++ listing in the book is ported to a Mojo package, `finrecipes`,
and every example in the book whose output is printed is checked by a test.

## Requirements

- [pixi](https://pixi.sh) (installs the Mojo toolchain for you)
- macOS on Apple silicon, or Linux x86-64

The project was written and tested against Mojo `1.3.0.dev2026111105`
(nightly). `pixi.lock` pins that version. Mojo is still changing quickly, so
a much newer compiler may need small syntax fixes; see
[docs/PORTING_NOTES.md](docs/PORTING_NOTES.md).

## Quick start

```sh
pixi install          # downloads Mojo into .pixi/
pixi run test         # runs all 29 test files (187 tests)
pixi run audit        # reruns every book example, rebuilds docs/AUDIT.md
```

## How the results compare with the book

[docs/AUDIT.md](docs/AUDIT.md) lists all 77 output boxes printed in the book
and compares each printed number with the port. Of 234 numbers, 232 match to
the printed digits, including every Monte Carlo result. The two that differ
are the Ho-Lee bond option prices of chapter 25, where the book's listing
reads past the end of a vector. One box (the Boost date demo of chapter 27)
has nothing to compare. The same report is available as
[docs/AUDIT.pdf](docs/AUDIT.pdf) (rebuilt with `python3 audit/make_pdf.py`,
which needs the `reportlab` package).

Run one test file:

```sh
pixi run mojo run -I . tests/test_black_scholes.mojo
```

Use the package from your own program (run it with `-I` pointing at this
folder):

```mojo
from finrecipes.black_scholes import option_price_call_black_scholes

def main():
    # S = 50, K = 50, r = 10%, sigma = 30%, six months
    print(option_price_call_black_scholes(50.0, 50.0, 0.10, 0.30, 0.50))  # 5.45325
```

## Book chapter to module

| Chapter | Topic | Module | Test file |
|---|---|---|---|
| 1 | Date class | `finrecipes/date.mojo` | `tests/test_date.mojo` |
| 2 | Matrix tools (Matlab examples) | `finrecipes/matrix.mojo` | `tests/test_matrix.mojo` |
| 3 | The value of time: PV, IRR | `finrecipes/cashflow.mojo` | `tests/test_cashflow.mojo` |
| 4 | Bond pricing, flat term structure | `finrecipes/bonds.mojo` | `tests/test_bonds.mojo` |
| 5 | Term structure of interest rates | `finrecipes/term_structure.mojo` | `tests/test_term_structure.mojo` |
| 6 | Mean variance frontier, CAPM measures | `finrecipes/mean_variance.mojo` | `tests/test_mean_variance.mojo` |
| 7 | Futures | `finrecipes/futures.mojo` | `tests/test_futures.mojo` |
| 8 | Binomial option pricing, basics | `finrecipes/binomial_basic.mojo` | `tests/test_binomial_basic.mojo` |
| 9 | Black Scholes formula, Greeks, implied volatility | `finrecipes/black_scholes.mojo` | `tests/test_black_scholes.mojo` |
| 10 | Warrants | `finrecipes/warrants.mojo` | `tests/test_warrants.mojo` |
| 11 | Extending Black Scholes | `finrecipes/black_scholes_extensions.mojo` | `tests/test_black_scholes_extensions.mojo` |
| 12 | Binomial approximations | `finrecipes/binomial.mojo` | `tests/test_binomial.mojo` |
| 13 | Finite differences | `finrecipes/finite_differences.mojo` | `tests/test_finite_differences.mojo` |
| 14 | Option pricing by simulation | `finrecipes/simulation.mojo` | `tests/test_simulation.mojo` |
| 15 | American option approximations | `finrecipes/american_approximations.mojo` | `tests/test_american_approximations.mojo` |
| 16 | Bermudan, Asian, lookback options | `finrecipes/exotics.mojo` | `tests/test_exotics.mojo` |
| 17 | Generic binomial pricing | `finrecipes/generic_binomial.mojo` | `tests/test_generic_binomial.mojo` |
| 18 | Trinomial trees | `finrecipes/trinomial.mojo` | `tests/test_trinomial.mojo` |
| 19 | Merton jump diffusion, Heston | `finrecipes/alternative_formulas.mojo` | `tests/test_alternative_formulas.mojo` |
| 20 | Bond options | `finrecipes/bond_options.mojo` | `tests/test_bond_options.mojo` |
| 21 | Credit risk (Merton model) | `finrecipes/credit_risk.mojo` | `tests/test_credit_risk.mojo` |
| 22 | Term structure models | `finrecipes/term_structure_models.mojo` | `tests/test_term_structure_models.mojo` |
| 23 | Rendleman and Bartter | `finrecipes/rendleman_bartter.mojo` | `tests/test_rendleman_bartter.mojo` |
| 24 | Interest rate trees, callable bond | `finrecipes/interest_rate_trees.mojo` | `tests/test_interest_rate_trees.mojo` |
| 25 | Ho and Lee term structure trees | `finrecipes/ho_lee.mojo` | `tests/test_ho_lee.mojo` |
| 26 | Vasicek bond option | `finrecipes/term_structure_derivatives.mojo` | `tests/test_term_structure_derivatives.mojo` |
| App. A | Normal distribution, random numbers | `finrecipes/normdist.mojo`, `finrecipes/random.mojo` | `tests/test_normdist.mojo` |
| App. C | Trivariate normal N3 | `finrecipes/normdist3.mojo` | `tests/test_normdist3.mojo` |

Chapter 27 (dates with the Boost library) and appendices B, D, E and F
contain no algorithms of their own.

## How the port differs from the C++

- **Names.** Routines keep the book's names in snake_case. C++ classes are
  CamelCase structs (`term_structure_class_flat` is `TermStructureFlat`).
  The overloaded bivariate `N(a, b, rho)` is `N2`; the trivariate one is `N3`.
- **Base classes are traits.** `term_structure_class` is the `TermStructure`
  trait; functions that take any term structure are generic over it.
- **Out-parameters are return values.** Routines that fill reference
  arguments in C++ return a struct, for example `Greeks`.
- **External libraries are replaced by code in this package.**
  - Newmat, IT++ and Matlab matrix code uses `finrecipes/matrix.mojo`.
    Where the book shows the same algorithm for two libraries it is ported once.
  - GSL numerical integration (Heston, N3) is replaced by hand-written
    adaptive Gauss-Kronrod quadrature with the book's tolerances.
- **Random numbers.** The book uses the C library `rand()`. By default this
  port uses Mojo's generator (`random_seed(n)`), whose sequences differ from
  the book's. Calling `random_seed_c_rand(1)` switches to an emulation of
  the GNU C library `rand()`, and the book's Monte Carlo results are then
  reproduced digit for digit (`tests/test_book_monte_carlo.mojo`). The
  generator state is held with `std.ffi._Global`, a private part of Mojo's
  standard library that may change in later versions.
- **Reserved words.** `lambda` arguments are named `lam` or `lambda_`.
- **Same name in two modules.** As in the book's overloads, `bonds_price`,
  `bonds_duration` and `bonds_convexity` exist in both `bonds.mojo` (flat
  rate) and `term_structure.mojo` (term structure object); `payoff_call` and
  `payoff_put` exist in both `simulation.mojo` and `generic_binomial.mojo`;
  the generic Monte Carlo routines exist in both `simulation.mojo` and
  `exotics.mojo`. Import with `as` if you need both in one file.

## Errors found in the book's listings

The port follows the printed code, so that the book's outputs are
reproduced, except where noted. Each item is documented in the docstring of
the routine concerned.

| Listing | Problem | What the port does |
|---|---|---|
| C++ Code 15.2 to 15.5 (Geske-Johnson put, Barone-Adesi Whaley call, Bjerksund-Stensland call and put) | The listings do not compute the chapter's formulas (cost of carry `b` passed where payout rate `q` is expected, interchanged terms). The printed Barone-Adesi Whaley output 5.74339 is an artefact; the published value for that example is 3.52. | Ported as printed under the book's names, plus `_corrected` versions that are tested against published values and a binomial tree. |
| C++ Code 25.4 (Ho-Lee call on a bond) | Reads one element past the end of a vector, so the printed prices 6.4857 and 6.35411 depend on a stray memory value. | Fills the missing payoff: 7.09845 and 6.95282. A separate test reproduces the book's numbers by substituting the stray value. |
| C++ Code 25.3 (Ho-Lee tree) | Hard-codes five tree levels and ignores `no_steps`. | Builds `no_steps` levels (the same tree for the book's example). |
| C++ Code 22.9 (Vasicek, `a == 0` branch) | Misplaced parenthesis gives a discount factor of 1/6 at time 0. | Corrected to the limit of the general branch. |
| C++ Code 8.3 (multiperiod binomial) | Steps terminal prices by `u*u`, which is only right when `d = 1/u`. | Ported as printed. |
| Chapter 11 futures option example | Output is labelled "call" but the program prices the put (0.851476). | Test asserts the put. |
| Chapter 24 callable bond | The lattice in the text (89.35) and the program output (89.2483) differ by one period in the first call date. | Both values are asserted. |
| C++ Code 22.1 (Nelson Siegel) | Uses `+ beta2*exp(-t/lambda)` where the usual form has a minus. | Ported as printed; the book's output depends on it. |

## Layout

```
finrecipes/        the Mojo package, one module per chapter
tests/             one test program per module
audit/             reruns every book example and builds docs/AUDIT.md
docs/              porting notes and the audit against the book
run_tests.sh       runs every test file, non-zero exit on failure
pixi.toml          toolchain and the `test` task
.github/workflows  CI: runs the tests on Linux and macOS
```

## Credits and licence

The algorithms, routine names and example values are from Bernt Arne
Ødegaard, *Financial Numerical Recipes in C++*. The book does not state a
licence for its source code. Check the terms on the author's home page
before publishing this repository publicly or adding a licence file.
