"""Turn the output of audit/book_examples.mojo into docs/AUDIT.md.

Usage: mojo run -I . audit/book_examples.mojo | python3 audit/make_table.py > docs/AUDIT.md
"""
import sys
from collections import OrderedDict


def decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


def fmt(x, dec):
    return f"{x:.{dec}f}"


TITLES = {
    1: ("Matlab", "Present value of cash flows at 10%"),
    2: ("C++", "Present value, discrete compounding"),
    3: ("Matlab", "Net present value and internal rate of return"),
    4: ("C++", "Present value and internal rate of return"),
    5: ("Matlab", "Converting between compounding conventions"),
    6: ("Matlab", "Bond price"),
    7: ("C++", "Bond price"),
    8: ("Matlab", "Bond price and yield"),
    9: ("C++", "Bond price and yield to maturity"),
    10: ("Matlab", "Duration and Macaulay duration"),
    11: ("C++", "Duration and Macaulay duration"),
    12: ("Matlab", "Duration and convexity"),
    13: ("C++", "Modified duration and convexity"),
    14: ("Matlab", "Duration-adjusted price change"),
    15: ("C++", "Bond with continuous compounding"),
    16: ("C++", "Term structure transformations"),
    17: ("C++", "Flat term structure class"),
    18: ("Matlab", "Discount factors from bond prices"),
    19: ("C++", "Linear interpolation of yields"),
    20: ("C++", "Interpolated term structure class"),
    21: ("C++", "Bond price, duration, convexity with a term structure"),
    22: ("Matlab", "Portfolio mean and standard deviation"),
    23: ("Matlab", "Minimum variance frontier portfolio"),
    24: ("C++", "Mean variance calculations"),
    25: ("C++", "Frontier portfolio for a given mean"),
    26: ("C++", "Futures price"),
    27: ("C++", "One and two period binomial call"),
    28: ("Matlab", "Black Scholes call"),
    29: ("C++", "Black Scholes call"),
    30: ("C++", "Black Scholes partial derivatives"),
    31: ("C++", "Implied volatility"),
    32: ("C++", "Warrant price"),
    33: ("C++", "European call with dividends"),
    34: ("C++", "American call with one dividend (Roll-Geske-Whaley)"),
    35: ("C++", "Futures option (Black)"),
    36: ("C++", "Currency option"),
    37: ("C++", "Perpetual American call"),
    38: ("C++", "Binomial European and American options"),
    39: ("Matlab", "Binomial American call and put"),
    40: ("C++", "Binomial hedge parameters"),
    41: ("C++", "Binomial with payouts and dividends"),
    42: ("C++", "Futures option, binomial"),
    43: ("C++", "Currency option, binomial"),
    44: ("C++", "Explicit finite differences"),
    45: ("C++", "Implicit finite differences"),
    46: ("Matlab", "Implicit finite differences, American put"),
    47: ("C++", "European call and put by simulation"),
    48: ("C++", "Delta by simulation"),
    49: ("C++", "Generic simulation pricing"),
    50: ("C++", "Control and antithetic variates"),
    51: ("C++", "Binary options by simulation"),
    52: ("C++", "Johnson approximation, American put"),
    53: ("C++", "Barone-Adesi Whaley approximation, American call"),
    54: ("C++", "Bermudan put"),
    55: ("C++", "Asian geometric average call"),
    56: ("C++", "Lookback call"),
    57: ("C++", "Path-dependent options by simulation"),
    58: ("C++", "Generic binomial call and put"),
    59: ("C++", "Generic binomial binary option"),
    60: ("C++", "Trinomial American put"),
    61: ("C++", "Merton jump diffusion"),
    62: ("C++", "Heston stochastic volatility"),
    63: ("C++", "Bond options"),
    64: ("C++", "Credit risk, Merton model"),
    65: ("C++", "Nelson Siegel term structure"),
    66: ("C++", "Cubic spline term structure"),
    67: ("C++", "Cox Ingersoll Ross term structure"),
    68: ("C++", "Vasicek term structure"),
    69: ("C++", "Rendleman Bartter bond option"),
    70: ("C++", "Interest rate tree"),
    71: ("C++", "Bond price in an interest rate tree"),
    72: ("C++", "Callable bond"),
    73: ("C++", "Ho-Lee bond option"),
    74: ("C++", "Vasicek bond option"),
    75: ("C++", "Boost date and time demo"),
    76: ("C++", "Cumulative normal, univariate and bivariate"),
    77: ("C++", "Random numbers"),
}

NOTES = [
    (73, "The book's C++ Code 25.4 fills T+1 of the T+2 option values at the last tree level and then reads one element past them, so the printed prices depend on a stray memory value. The port fills the missing payoff. `tests/test_ho_lee.mojo` shows that the printed 6.4857 and 6.35411 are exactly what results when the stray value is 0.5."),
    (53, "The port reproduces the printed 5.74339, but the listing passes the cost of carry `b` where the payout rate is expected, and its Newton iteration does not converge for this example. The published Barone-Adesi Whaley value is 3.52, which `option_price_american_call_approximated_baw_corrected` returns."),
    (35, "The book labels the output a call, but the program calls `futures_option_price_put_european_black`. The printed 0.851476 is the put price, and the port's put matches it."),
    (75, "Demonstrates the Boost C++ date library and prints the clock time of the run. Nothing to port or compare."),
    (12, "The Matlab convexity formula uses t squared and gives 6.6272; C++ Code 4.6 (box 13) uses t(t+1) and gives 8.93248. Both are reproduced; the Matlab value is computed inline in the audit program, because the package ports the C++ routine."),
    (5, "The book has no routine for these two conversions; they are evaluated inline in the audit program."),
    (18, "The discount factors come from `mat_solve`; the conversion to annual rates is evaluated inline."),
]

rows = []
for line in sys.stdin:
    p = line.rstrip("\n").split("|")
    if p[0] == "ROW":
        _, box, ch, label, book, port = p
        dec = decimals(book)
        diff = float(port) - float(book)
        unit = 10.0 ** (-dec)
        if abs(diff) <= 0.5 * unit * (1 + 1e-9) + 1e-13:
            res = "match"
        elif abs(diff) <= unit * (1 + 1e-9):
            res = "last digit"
        else:
            res = "DIFFERS"
        rows.append((int(box), ch, label, book, fmt(float(port), dec + 2), f"{diff:+.1e}", res, "det"))
    elif p[0] == "NA":
        _, box, ch, label, book, reason = p
        rows.append((int(box), ch, label, book, "n/a", reason, "not comparable", "na"))

boxes = OrderedDict()
for r in rows:
    boxes.setdefault(r[0], []).append(r)


def box_status(rs):
    res = {r[6] for r in rs}
    if "DIFFERS" in res or "last digit" in res:
        return "differs"
    if res == {"not comparable"}:
        return "not comparable"
    return "match"


counts = {}
for b, rs in boxes.items():
    counts[box_status(rs)] = counts.get(box_status(rs), 0) + 1
nvals = {}
for r in rows:
    nvals[r[6]] = nvals.get(r[6], 0) + 1

out = []
w = out.append
w("# Audit: book examples against the Mojo port")
w("")
w("Generated by `sh audit/run_audit.sh`. Do not edit by hand.")
w("")
w("The book prints 77 output boxes (\"Output from C++ program\" and \"Output from")
w("Matlab program\"). `audit/book_examples.mojo` reruns the program of every box")
w("with the `finrecipes` package and this table compares each printed number")
w("with the port's result. Inputs that Matlab merely echoes back are left out.")
w("")
w("The Monte Carlo examples (boxes 47 to 51, 57 and 77) are run with the package's")
w("emulation of the GNU C library `rand()`, seeded with 1 at the start of each box,")
w("which is the random sequence the book's programs used. They are therefore")
w("compared digit for digit, like every other number.")
w("")
w("## How to read the result column")
w("")
w("- **match**: the port agrees with the book to within half a unit of the last printed digit.")
w("- **last digit**: the port differs by at most one unit in the last printed digit.")
w("- **DIFFERS**: the port gives a different number.")
w("- **not comparable**: nothing in the port to compare with (reason given).")
w("")
w("## Summary")
w("")
w(f"Boxes: {len(boxes)}. Printed numbers compared: {sum(1 for r in rows if r[7] != 'na')}.")
w("")
w("| Box result | Boxes |")
w("|---|---|")
for k in ["match", "differs", "not comparable"]:
    if k in counts:
        w(f"| {k} | {counts[k]} |")
w("")
w("| Number result | Numbers |")
w("|---|---|")
for k in ["match", "last digit", "DIFFERS", "not comparable"]:
    if k in nvals:
        w(f"| {k} | {nvals[k]} |")
w("")
bad = [r for r in rows if r[6] in ("DIFFERS", "last digit")]
w("## Numbers that do not match")
w("")
if bad:
    w("| Box | Ch. | Quantity | Book | Port | Difference | Result |")
    w("|---|---|---|---|---|---|---|")
    for r in bad:
        w(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")
else:
    w("None.")
w("")
w("## The 77 boxes")
w("")
w("| Box | Ch. | Source | Example | Numbers compared | Matching | Result |")
w("|---|---|---|---|---|---|---|")
for b, rs in boxes.items():
    lang, title = TITLES[b]
    comp = [r for r in rs if r[7] != "na"]
    ok = sum(1 for r in comp if r[6] == "match")
    w(f"| {b} | {rs[0][1]} | {lang} | {title} | {len(comp)} | {ok} | {box_status(rs)} |")
w("")
w("## Notes on individual boxes")
w("")
for b, text in NOTES:
    w(f"- **Box {b}.** {text}")
w("")
w("## Every number")
w("")
w("| Box | Ch. | Quantity | Book | Port | Difference / note | Result |")
w("|---|---|---|---|---|---|---|")
for r in rows:
    w(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")
print("\n".join(out))
