#!/usr/bin/env bash
# =============================================================================
# verify_sandbox.sh -- Verify a SandboxFusion sandbox server is ready for SR2AM
#
# Checks:
#   1. Server reachable (basic code execution)
#   2. Critical Python packages importable
#   3. Computation produces correct output
#
# Usage: bash scripts/sandbox/verify_sandbox.sh [--host HOST] [--port PORT]
# =============================================================================

set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-8080}"

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --help) echo "Usage: bash scripts/sandbox/verify_sandbox.sh [--host HOST] [--port PORT]"; exit 0 ;;
        *) shift ;;
    esac
done

API_URL="http://${HOST}:${PORT}/run_code"

run_code() {
    local code="$1"
    local timeout="${2:-30}"
    curl -s "$API_URL" \
        -H 'Content-Type: application/json' \
        --data-raw "{\"code\": $(printf '%s' "$code" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'), \"language\": \"python\", \"run_timeout\": $timeout}"
}

passed=0
failed=0
total=0

check() {
    local name="$1"
    local code="$2"
    local expect="${3:-}"
    total=$((total + 1))

    result=$(run_code "$code" 30 2>/dev/null || echo "CURL_ERROR")

    if echo "$result" | grep -q "CURL_ERROR"; then
        echo "  FAIL  $name (connection error)"
        failed=$((failed + 1))
        return
    fi

    if echo "$result" | grep -q '"status":"Failed"\|"status":"SandboxError"'; then
        echo "  FAIL  $name (execution error)"
        failed=$((failed + 1))
        return
    fi

    if [[ -n "$expect" ]]; then
        if echo "$result" | grep -q "$expect"; then
            echo "  PASS  $name"
            passed=$((passed + 1))
        else
            echo "  FAIL  $name (unexpected output)"
            failed=$((failed + 1))
        fi
    else
        echo "  PASS  $name"
        passed=$((passed + 1))
    fi
}

echo "Verifying sandbox at $API_URL"
echo "============================================================"

# --- Basic connectivity ---
echo ""
echo "Connectivity:"
check "basic execution" 'print("sandbox_ok")' "sandbox_ok"

# --- Critical packages (used frequently in SR2AM agent traces) ---
echo ""
echo "Core scientific packages:"
check "numpy"       'import numpy; print(numpy.__version__)'
check "pandas"      'import pandas; print(pandas.__version__)'
check "scipy"       'import scipy; print(scipy.__version__)'
check "sympy"       'import sympy; print(sympy.__version__)'
check "matplotlib"  'import matplotlib; print(matplotlib.__version__)'
check "sklearn"     'import sklearn; print(sklearn.__version__)'
check "statsmodels" 'import statsmodels; print(statsmodels.__version__)'

echo ""
echo "Math and logic:"
check "z3-solver"   'import z3; print(z3.get_version_string())'
check "networkx"    'import networkx; print(networkx.__version__)'
check "PuLP"        'import pulp; print(pulp.__version__)'
check "ortools"     'from ortools.sat.python import cp_model; print("ok")' "ok"
check "cvxpy"       'import cvxpy; print(cvxpy.__version__)'

echo ""
echo "Domain-specific:"
check "chess"       'import chess; print(chess.__version__)'
check "rdkit"       'from rdkit import Chem; print(Chem.MolToSmiles(Chem.MolFromSmiles("C")))' "C"
check "beautifulsoup4" 'from bs4 import BeautifulSoup; print("ok")' "ok"
check "requests"    'import requests; print(requests.__version__)'
check "lxml"        'import lxml; print(lxml.__version__)'
check "PIL"         'from PIL import Image; print("ok")' "ok"
check "pdfplumber"  'import pdfplumber; print("ok")' "ok"

echo ""
echo "Data and serialization:"
check "pyarrow"     'import pyarrow; print(pyarrow.__version__)'
check "dask"        'import dask; print(dask.__version__)'
check "xarray"      'import xarray; print(xarray.__version__)'

echo ""
echo "Computation correctness:"
check "arithmetic" \
    'import numpy as np; print(int(np.sum(np.arange(100))))' \
    "4950"
check "sympy solve" \
    'from sympy import symbols, solve; x = symbols("x"); print(solve(x**2 - 4, x))' \
    "\-2.*2"

# --- Summary ---
echo ""
echo "============================================================"
echo "Results: $passed/$total passed, $failed failed"
echo "============================================================"

if [[ $failed -gt 0 ]]; then
    echo "[WARN] Some checks failed. Run setup_sandbox.sh to install missing packages."
    exit 1
else
    echo "[OK] Sandbox is ready for SR2AM inference."
    exit 0
fi
