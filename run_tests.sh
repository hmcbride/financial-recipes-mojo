#!/bin/sh
# Run every test file in tests/. Exits non-zero if any file fails.
# Usage: pixi run test        (or: sh run_tests.sh)
cd "$(dirname "$0")" || exit 1
status=0
for f in tests/test_*.mojo; do
    echo "== $f"
    if ! mojo run -I . "$f"; then
        status=1
    fi
done
if [ "$status" -eq 0 ]; then
    echo "ALL TEST FILES PASSED"
else
    echo "SOME TEST FILES FAILED"
fi
exit $status
