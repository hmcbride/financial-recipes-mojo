"""Chapter 1: on C++ and programming (C++ Code 1.1 - 1.4, Header file 1.1).

The C++ `date` class becomes the `Date` struct. C++ keeps the data in the
protected members `day_`, `month_`, `year_` and reads them with `day()`,
`month()`, `year()`; the struct does the same.

Mojo has no `++` and `--` operators, so the four iteration operators of
C++ Code 1.4 are methods:

    ++d   d.pre_increment()     d++   d.post_increment()
    --d   d.pre_decrement()     d--   d.post_decrement()
"""

from std.math import exp, log


def power(x: Float64, n: Float64) -> Float64:
    """A simple power function, x^n = exp(n ln x) (C++ Code 1.1)."""
    var p = exp(n * log(x))
    return p


struct Date(ImplicitlyCopyable, Movable):
    """A trivial date class (Header file 1.1, C++ Code 1.2 - 1.4)."""

    var year_: Int
    var month_: Int
    var day_: Int

    def __init__(out self):
        """The default date, 0/0/0, which is not valid."""
        self.year_ = 0
        self.month_ = 0
        self.day_ = 0

    def __init__(out self, day: Int, month: Int, year: Int):
        self.day_ = day
        self.month_ = month
        self.year_ = year

    # Basic operations (C++ Code 1.2)

    def day(self) -> Int:
        return self.day_

    def month(self) -> Int:
        return self.month_

    def year(self) -> Int:
        return self.year_

    def set_day(mut self, day: Int):
        self.day_ = day

    def set_month(mut self, month: Int):
        self.month_ = month

    def set_year(mut self, year: Int):
        self.year_ = year

    def valid(self) -> Bool:
        """Check whether the date is valid (C++ Code 1.2).

        As in the book there is no leap year check: 29 February is accepted
        in any year.
        """
        if self.year_ < 0:
            return False
        if self.month_ > 12 or self.month_ < 1:
            return False
        if self.day_ > 31 or self.day_ < 1:
            return False
        if self.day_ == 31 and (
            self.month_ == 2 or self.month_ == 4 or self.month_ == 6 or self.month_ == 9 or self.month_ == 11
        ):
            return False
        if self.day_ == 30 and self.month_ == 2:
            return False
        return True

    # Comparison operators (C++ Code 1.3)

    def __eq__(self, other: Date) -> Bool:
        """Equality. If either date is not valid the answer is False."""
        if not (self.valid() and other.valid()):
            return False
        return self.day_ == other.day_ and self.month_ == other.month_ and self.year_ == other.year_

    def __lt__(self, other: Date) -> Bool:
        """Earlier than. If either date is not valid the answer is False."""
        if not (self.valid() and other.valid()):
            return False
        if self.year_ == other.year_:  # same year
            if self.month_ == other.month_:  # same month
                return self.day_ < other.day_
            else:
                return self.month_ < other.month_
        else:  # different year
            return self.year_ < other.year_

    # remaining operators defined in terms of the above

    def __le__(self, other: Date) -> Bool:
        if self == other:
            return True
        return self < other

    def __ge__(self, other: Date) -> Bool:
        if self == other:
            return True
        return self > other

    def __gt__(self, other: Date) -> Bool:
        return not (self <= other)

    def __ne__(self, other: Date) -> Bool:
        return not (self == other)

    # Iteration operators (C++ Code 1.4)

    def post_increment(mut self) -> Date:
        """Postfix `d++`: move to the next date, return the old value."""
        var d = self.copy()
        self = next_date(d)
        return d

    def pre_increment(mut self) -> Date:
        """Prefix `++d`: move to the next date, return the new value."""
        self = next_date(self)
        return self.copy()

    def post_decrement(mut self) -> Date:
        """Postfix `d--`: move to the previous date, return the old value."""
        var d = self.copy()
        self = previous_date(self)
        return d

    def pre_decrement(mut self) -> Date:
        """Prefix `--d`: move to the previous date, return the new value."""
        self = previous_date(self)
        return self.copy()


def next_date(d: Date) -> Date:
    """The date following d (C++ Code 1.4). An invalid d gives the default date."""
    if not d.valid():
        return Date()
    var ndat = Date(d.day() + 1, d.month(), d.year())  # first try adding a day
    if ndat.valid():
        return ndat
    ndat = Date(1, d.month() + 1, d.year())  # then try adding a month
    if ndat.valid():
        return ndat
    ndat = Date(1, 1, d.year() + 1)  # must be next year
    return ndat


def previous_date(d: Date) -> Date:
    """The date preceding d (C++ Code 1.4). An invalid d gives the default date."""
    if not d.valid():
        return Date()  # return the default date
    var pdat = Date(d.day() - 1, d.month(), d.year())  # try same month
    if pdat.valid():
        return pdat
    pdat = Date(31, d.month() - 1, d.year())  # try previous month
    if pdat.valid():
        return pdat
    pdat = Date(30, d.month() - 1, d.year())
    if pdat.valid():
        return pdat
    pdat = Date(29, d.month() - 1, d.year())
    if pdat.valid():
        return pdat
    pdat = Date(28, d.month() - 1, d.year())
    if pdat.valid():
        return pdat
    pdat = Date(31, 12, d.year() - 1)  # try previous year
    return pdat
