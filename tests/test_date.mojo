from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_true, assert_false
from finrecipes.date import power, Date, next_date, previous_date


def assert_date(d: Date, day: Int, month: Int, year: Int) raises:
    assert_equal(d.day(), day)
    assert_equal(d.month(), month)
    assert_equal(d.year(), year)


def test_power() raises:
    # Book output, C++ Code 1.1: 2^1 = 2, 2^2 = 4, 2^3 = 8, 2^4 = 16, 2^5 = 32
    var expected: List[Float64] = [2.0, 4.0, 8.0, 16.0, 32.0]
    for n in range(1, 6):
        assert_almost_equal(power(2.0, Float64(n)), expected[n - 1], atol=1e-10)


def test_default_date_is_invalid() raises:
    var d = Date()
    assert_date(d, 0, 0, 0)
    assert_false(d.valid())


def test_accessors_and_setters() raises:
    var d = Date(12, 8, 2003)
    assert_date(d, 12, 8, 2003)
    d.set_day(1)
    d.set_month(2)
    d.set_year(1995)
    assert_date(d, 1, 2, 1995)


def test_valid() raises:
    assert_true(Date(1, 1, 2001).valid())
    assert_true(Date(31, 12, 1999).valid())
    assert_true(Date(1, 1, 0).valid())
    assert_false(Date(1, 1, -1).valid())  # negative year
    assert_false(Date(1, 0, 2001).valid())  # month out of range
    assert_false(Date(1, 13, 2001).valid())
    assert_false(Date(0, 1, 2001).valid())  # day out of range
    assert_false(Date(32, 1, 2001).valid())
    # months with 30 days
    assert_false(Date(31, 4, 2001).valid())
    assert_false(Date(31, 6, 2001).valid())
    assert_false(Date(31, 9, 2001).valid())
    assert_false(Date(31, 11, 2001).valid())
    assert_true(Date(30, 4, 2001).valid())
    # months with 31 days
    assert_true(Date(31, 1, 2001).valid())
    assert_true(Date(31, 7, 2001).valid())
    assert_true(Date(31, 8, 2001).valid())
    # february: the book accepts 29 feb in any year, leap year or not
    assert_false(Date(31, 2, 2001).valid())
    assert_false(Date(30, 2, 2001).valid())
    assert_true(Date(29, 2, 2001).valid())
    assert_true(Date(28, 2, 2001).valid())


def test_comparisons() raises:
    var a = Date(12, 8, 2003)
    var b = Date(12, 8, 2003)
    var later_day = Date(13, 8, 2003)
    var later_month = Date(1, 9, 2003)
    var later_year = Date(1, 1, 2004)
    assert_true(a == b)
    assert_false(a != b)
    assert_true(a <= b)
    assert_true(a >= b)
    assert_false(a < b)
    assert_false(a > b)

    assert_true(a < later_day)
    assert_true(a < later_month)
    assert_true(a < later_year)
    assert_true(later_month < later_year)
    assert_false(later_day < a)
    assert_false(later_year < later_month)

    assert_true(a != later_day)
    assert_true(a <= later_day)
    assert_false(a >= later_day)
    assert_true(later_day > a)
    assert_true(later_day >= a)
    assert_false(later_day <= a)


def test_comparisons_with_invalid_date() raises:
    # C++ Code 1.3: == and < are false when either date is not valid.
    # The other operators are defined from those two, so != and > are true.
    var good = Date(1, 1, 2001)
    var bad = Date()
    assert_false(good == bad)
    assert_false(bad == bad)
    assert_false(good < bad)
    assert_false(bad < good)
    assert_false(good <= bad)
    assert_true(good != bad)
    assert_true(good > bad)
    assert_true(good >= bad)


def test_next_date() raises:
    assert_date(next_date(Date(1, 1, 1995)), 2, 1, 1995)
    assert_date(next_date(Date(31, 1, 1995)), 1, 2, 1995)  # month end, 31 days
    assert_date(next_date(Date(30, 4, 1995)), 1, 5, 1995)  # month end, 30 days
    assert_date(next_date(Date(28, 2, 1995)), 29, 2, 1995)  # no leap year check
    assert_date(next_date(Date(29, 2, 1995)), 1, 3, 1995)
    assert_date(next_date(Date(31, 12, 1995)), 1, 1, 1996)  # year end
    assert_date(next_date(Date()), 0, 0, 0)  # invalid gives the default date


def test_previous_date() raises:
    assert_date(previous_date(Date(2, 1, 1995)), 1, 1, 1995)
    assert_date(previous_date(Date(1, 2, 1995)), 31, 1, 1995)  # back to a 31 day month
    assert_date(previous_date(Date(1, 5, 1995)), 30, 4, 1995)  # back to a 30 day month
    assert_date(previous_date(Date(1, 3, 1995)), 29, 2, 1995)  # no leap year check
    assert_date(previous_date(Date(1, 1, 1996)), 31, 12, 1995)  # year start
    assert_date(previous_date(Date()), 0, 0, 0)


def test_increment() raises:
    # Book text, section 1.2.2: date d(1,1,1995); d++; gives 2 jan 95
    var d = Date(1, 1, 1995)
    var old = d.post_increment()
    assert_date(old, 1, 1, 1995)
    assert_date(d, 2, 1, 1995)
    # Book text, section 1.3.1: date d(1,1,2001); ++d; gives 2 january 2001
    var e = Date(1, 1, 2001)
    var new = e.pre_increment()
    assert_date(new, 2, 1, 2001)
    assert_date(e, 2, 1, 2001)


def test_decrement() raises:
    var d = Date(1, 1, 1995)
    var old = d.post_decrement()
    assert_date(old, 1, 1, 1995)
    assert_date(d, 31, 12, 1994)
    var e = Date(1, 3, 2001)
    var new = e.pre_decrement()
    assert_date(new, 29, 2, 2001)
    assert_date(e, 29, 2, 2001)


def test_walk_through_a_year() raises:
    # 1 jan to 31 dec with the book's calendar (february always has 29 days)
    # is 366 dates, so 365 steps forward and 365 steps back again.
    var d = Date(1, 1, 2003)
    var steps = 0
    while d < Date(31, 12, 2003):
        var prev = d.post_increment()
        assert_true(prev < d)
        assert_true(d.valid())
        steps += 1
    assert_equal(steps, 365)
    assert_date(d, 31, 12, 2003)
    for _ in range(365):
        var prev = d.pre_decrement()
        assert_true(prev.valid())
    assert_date(d, 1, 1, 2003)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
