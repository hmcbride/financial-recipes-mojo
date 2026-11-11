#!/bin/sh
# Rerun every example in the book against the port and rebuild docs/AUDIT.md.
# Usage: pixi run audit      (or: sh audit/run_audit.sh)
cd "$(dirname "$0")/.." || exit 1
mojo run -I . audit/book_examples.mojo > audit/audit_output.txt || exit 1
python3 audit/make_table.py < audit/audit_output.txt > docs/AUDIT.md || exit 1
echo "wrote docs/AUDIT.md"
