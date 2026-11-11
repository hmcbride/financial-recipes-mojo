"""Appendix A: pseudorandom numbers (C++ Code A.4 and A.5).

The book draws its uniform numbers from the C library `rand()`. Two
generators are available here:

- Mojo's standard library generator (the default). Call `random_seed` for
  reproducible runs. Its sequences differ from the book's printed output.
- An emulation of the GNU C library `rand()`, which is what produced the
  simulation results printed in the book. Call `random_seed_c_rand(1)` to
  get the sequence a freshly started C++ program sees; the Monte Carlo
  examples of the book are then reproduced digit for digit (see
  docs/AUDIT.md).

Mojo has no module-level variables, so the generator state is kept with
`std.ffi._Global`, a private part of the standard library that may change
between Mojo versions.
"""

from std.ffi import _Global
from std.math import log, sqrt
from std.random import random_float64, seed

comptime RAND_MAX: Float64 = 2147483647.0


struct _CRandState(Movable):
    """State of the emulated GNU C library `rand()` (the TYPE_3 additive generator)."""

    var active: Bool  # True: random_uniform_0_1 draws from this generator
    var k: Int  # index of the next number in the recurrence
    var buf: List[Int]  # the last 34 numbers of the recurrence, as a ring

    def __init__(out self):
        self.active = False
        self.k = 0
        self.buf = List[Int](length=34, fill=0)

    def reseed(mut self, s: Int):
        """What srand(s) does: fill the table, then discard the first 310 numbers."""
        var sd = s
        if sd == 0:
            sd = 1
        self.buf[0] = sd
        for i in range(1, 31):
            var hi = self.buf[i - 1] // 127773
            var lo = self.buf[i - 1] % 127773
            var word = 16807 * lo - 2836 * hi
            if word < 0:
                word += 2147483647
            self.buf[i] = word
        for i in range(31, 34):
            self.buf[i] = self.buf[i - 31]
        self.k = 34
        for _ in range(310):
            _ = self.step()

    def step(mut self) -> Int:
        """Next 32 bit number of the recurrence r[k] = r[k-31] + r[k-3]."""
        var v = (self.buf[(self.k - 31) % 34] + self.buf[(self.k - 3) % 34]) & 0xFFFFFFFF
        self.buf[self.k % 34] = v
        self.k += 1
        return v

    def rand(mut self) -> Int:
        """What rand() returns: a number between 0 and RAND_MAX."""
        return self.step() >> 1


def _init_c_rand_state() -> _CRandState:
    return _CRandState()


comptime _C_RAND = _Global["finrecipes_c_rand_state", _init_c_rand_state]


def random_seed(s: Int):
    """Use Mojo's generator, seeded with s, so that simulations are reproducible."""
    try:
        var p = _C_RAND.get_or_create_ptr()
        p[].active = False
    except:
        pass
    seed(s)


def random_seed_c_rand(s: Int):
    """Use the emulated GNU C library rand(), seeded as srand(s) would.

    A C program that never calls srand starts with seed 1, so
    `random_seed_c_rand(1)` gives the sequence behind the book's output.
    """
    try:
        var p = _C_RAND.get_or_create_ptr()
        p[].reseed(s)
        p[].active = True
    except:
        pass


def random_uniform_0_1() -> Float64:
    """Pseudorandom number from a uniform [0,1) distribution (C++ Code A.4).

    With the emulated C generator this is rand()/RAND_MAX exactly as in the
    book, which can (very rarely) return 1.0.
    """
    try:
        var p = _C_RAND.get_or_create_ptr()
        if p[].active:
            return Float64(p[].rand()) / RAND_MAX
    except:
        pass
    return random_float64()


def random_normal() -> Float64:
    """Pseudorandom N(0,1) number using the polar method (C++ Code A.5)."""
    var V1: Float64 = 0.0
    var S: Float64 = 2.0
    while S >= 1.0 or S == 0.0:
        var U1 = random_uniform_0_1()
        var U2 = random_uniform_0_1()
        V1 = 2.0 * U1 - 1.0
        var V2 = 2.0 * U2 - 1.0
        S = V1 * V1 + V2 * V2
    return V1 * sqrt((-2.0 * log(S)) / S)
