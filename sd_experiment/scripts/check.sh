#!/usr/bin/env bash
#
# Run every offline validator in the harness.  No GPU required.
# Exits non-zero on the first failure so CI-style use works.
#
# Usage:
#     bash scripts/check.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

# Make package imports resolve when tests are run as scripts.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Phase B offline checks ==="
python3 tests/phase_b_offline_check.py

echo
echo "=== Phase C offline checks ==="
python3 tests/phase_c_offline_check.py

echo
echo "=== Phase D offline checks ==="
python3 tests/phase_d_offline_check.py

echo
echo "=== CLI --help smoke ==="
python3 -m runner.engine_runner --help > /dev/null
python3 tests/phase_b_smoke.py --help > /dev/null
python3 scripts/analyze_results.py --help > /dev/null
echo "[ok] all --help entry points parse"

echo
echo "All offline checks passed."
