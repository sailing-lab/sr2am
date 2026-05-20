#!/usr/bin/env bash
# =============================================================================
# setup_sandbox.sh -- Pull, configure, and verify a SandboxFusion sandbox server
#
# End-to-end setup:
#   1. Pull SandboxFusion Docker image via enroot
#   2. Create and start the container
#   3. Install Python packages from sandbox-requirements.txt via the /run_code API
#   4. Verify key packages are importable
#   5. Optionally export the configured container as a .sqsh for reuse
#
# Requirements: enroot, curl, jq (optional, for prettier output)
# =============================================================================

set -euo pipefail

# ========================= Defaults ==========================================
DOCKER_IMAGE="docker://varad0309/code_sandbox:server_unsecure"
SQSH_FILE="varad0309+code_sandbox+server_unsecure.sqsh"
CONTAINER_NAME="sandbox-server"
HOST="localhost"
PORT=8080
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/sandbox-requirements.txt"
SKIP_PULL=false
SKIP_INSTALL=false
EXPORT_FILE=""
CMD="make run-online"
LOG_DIR="${SCRIPT_DIR}/../../sandbox_logs"
# =============================================================================

usage() {
    cat <<'EOF'
Usage: bash scripts/sandbox/setup_sandbox.sh [OPTIONS]

Sets up a SandboxFusion code sandbox server with all required Python packages.

Steps:
  1. Pull Docker image via enroot import
  2. Create and start the Enroot container (runs SandboxFusion)
  3. Install packages from sandbox-requirements.txt via the /run_code API
  4. Verify key packages are importable
  5. (Optional) Export container as .sqsh for reuse with run_sandbox_retry.sh

Options:
  --docker-image IMAGE   Docker image to pull (default: docker://varad0309/code_sandbox:server_unsecure)
  --sqsh-file FILE       Local .sqsh filename (default: varad0309+code_sandbox+server_unsecure.sqsh)
  --container-name NAME  Enroot container name (default: sandbox-server)
  --host HOST            Server host for API calls (default: localhost)
  --port PORT            Server port (default: 8080)
  --requirements FILE    Requirements file (default: scripts/sandbox/sandbox-requirements.txt)
  --skip-pull            Skip enroot import (image .sqsh already exists)
  --skip-install         Skip pip install (packages already installed)
  --export FILE          Export configured container to .sqsh after setup
  --help                 Show this help message

Workflow:
  # First-time setup (pull, install, export)
  bash scripts/sandbox/setup_sandbox.sh --export sr2am_sandbox.sqsh

  # Re-install into existing container (image already pulled)
  bash scripts/sandbox/setup_sandbox.sh --skip-pull

  # Install packages into a remote server (already running)
  bash scripts/sandbox/setup_sandbox.sh --skip-pull --host SANDBOX_HOST

  # Verify only (packages already installed)
  bash scripts/sandbox/verify_sandbox.sh --host SANDBOX_HOST
EOF
    exit 0
}

# ========================= Parse Arguments ===================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --docker-image)     DOCKER_IMAGE="$2"; shift 2 ;;
        --sqsh-file)        SQSH_FILE="$2"; shift 2 ;;
        --container-name)   CONTAINER_NAME="$2"; shift 2 ;;
        --host)             HOST="$2"; shift 2 ;;
        --port)             PORT="$2"; shift 2 ;;
        --requirements)     REQUIREMENTS="$2"; shift 2 ;;
        --skip-pull)        SKIP_PULL=true; shift ;;
        --skip-install)     SKIP_INSTALL=true; shift ;;
        --export)           EXPORT_FILE="$2"; shift 2 ;;
        --help)             usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

API_URL="http://${HOST}:${PORT}/run_code"

# ========================= Helper Functions ==================================
run_code() {
    # Run code on the sandbox server. Args: code, language (default: python), timeout (default: 30)
    local code="$1"
    local language="${2:-python}"
    local timeout="${3:-30}"
    curl -s "$API_URL" \
        -H 'Content-Type: application/json' \
        --data-raw "{\"code\": $(printf '%s' "$code" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'), \"language\": \"$language\", \"run_timeout\": $timeout}"
}

wait_for_server() {
    local max_wait="${1:-300}"
    local interval=5
    local elapsed=0
    echo "[INFO] Waiting for sandbox server at $API_URL ..."
    while [[ $elapsed -lt $max_wait ]]; do
        if run_code 'print("ok")' python 5 2>/dev/null | grep -q "ok"; then
            echo "[INFO] Server is ready (took ${elapsed}s)"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    echo "[ERROR] Server not ready after ${max_wait}s"
    return 1
}

# ========================= Step 1: Pull Image ================================
if [[ "$SKIP_PULL" == false ]]; then
    echo "============================================================"
    echo "Step 1: Pulling SandboxFusion image"
    echo "============================================================"
    if [[ -f "$SQSH_FILE" ]]; then
        echo "[INFO] $SQSH_FILE already exists, skipping pull"
    else
        echo "[INFO] enroot import -o $SQSH_FILE $DOCKER_IMAGE"
        enroot import -o "$SQSH_FILE" "$DOCKER_IMAGE"
    fi
else
    echo "[INFO] Skipping image pull (--skip-pull)"
fi

# ========================= Step 2: Start Container ===========================
if [[ "$HOST" == "localhost" || "$HOST" == "127.0.0.1" ]]; then
    echo "============================================================"
    echo "Step 2: Starting SandboxFusion container"
    echo "============================================================"

    # Remove existing container if present
    if enroot list 2>/dev/null | awk '{print $1}' | grep -Fxq "$CONTAINER_NAME"; then
        echo "[INFO] Removing existing container: $CONTAINER_NAME"
        enroot remove -f "$CONTAINER_NAME" || true
    fi

    echo "[INFO] Creating container from $SQSH_FILE"
    enroot create --name "$CONTAINER_NAME" "$SQSH_FILE"

    mkdir -p "$LOG_DIR"
    local_log="${LOG_DIR}/setup_sandbox_$(date +%Y%m%d_%H%M%S).log"
    echo "[INFO] Starting container (log: $local_log)"
    enroot start --rw "$CONTAINER_NAME" bash -lc "$CMD" > "$local_log" 2>&1 &
    SANDBOX_PID=$!

    # Clean up on exit
    cleanup() {
        if [[ -n "${SANDBOX_PID:-}" ]]; then
            echo "[INFO] Stopping sandbox (PID: $SANDBOX_PID)"
            kill "$SANDBOX_PID" 2>/dev/null || true
            wait "$SANDBOX_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM

    wait_for_server 300
else
    echo "[INFO] Using remote server at $HOST:$PORT (skipping container start)"
    echo "[INFO] Checking server connectivity..."
    wait_for_server 30
fi

# ========================= Step 3: Install Packages ==========================
if [[ "$SKIP_INSTALL" == false ]]; then
    echo "============================================================"
    echo "Step 3: Installing packages from $REQUIREMENTS"
    echo "============================================================"

    if [[ ! -f "$REQUIREMENTS" ]]; then
        echo "[ERROR] Requirements file not found: $REQUIREMENTS"
        exit 1
    fi

    total=$(grep -cve '^\s*$' "$REQUIREMENTS" | grep -cve '^\s*#' || wc -l < "$REQUIREMENTS")
    installed=0
    failed=0
    failed_pkgs=()

    while IFS= read -r pkg || [[ -n "$pkg" ]]; do
        # Skip empty lines and comments
        pkg="$(echo "$pkg" | xargs)"
        [[ -z "$pkg" ]] && continue
        [[ "$pkg" == \#* ]] && continue

        installed=$((installed + 1))
        echo -ne "\r[INFO] Installing ($installed): $pkg                    "

        result=$(run_code "/root/miniconda3/envs/sandbox-runtime/bin/pip install $pkg" bash 300 2>/dev/null || echo "CURL_ERROR")

        if echo "$result" | grep -q '"status":"Failed"\|"status":"SandboxError"\|CURL_ERROR'; then
            failed=$((failed + 1))
            failed_pkgs+=("$pkg")
            echo ""
            echo "[WARN] Failed to install: $pkg"
        fi
    done < "$REQUIREMENTS"

    echo ""
    echo "[INFO] Installation complete: $installed packages attempted, $failed failed"
    if [[ $failed -gt 0 ]]; then
        echo "[WARN] Failed packages: ${failed_pkgs[*]}"
    fi
else
    echo "[INFO] Skipping package installation (--skip-install)"
fi

# ========================= Step 4: Verify ====================================
echo "============================================================"
echo "Step 4: Verifying sandbox"
echo "============================================================"
bash "${SCRIPT_DIR}/verify_sandbox.sh" --host "$HOST" --port "$PORT"

# ========================= Step 5: Export (Optional) =========================
if [[ -n "$EXPORT_FILE" ]]; then
    echo "============================================================"
    echo "Step 5: Exporting container to $EXPORT_FILE"
    echo "============================================================"
    if [[ "$HOST" != "localhost" && "$HOST" != "127.0.0.1" ]]; then
        echo "[WARN] Cannot export a remote container. Export only works for local containers."
    else
        # Stop the server before export
        if [[ -n "${SANDBOX_PID:-}" ]]; then
            echo "[INFO] Stopping sandbox for export..."
            kill "$SANDBOX_PID" 2>/dev/null || true
            wait "$SANDBOX_PID" 2>/dev/null || true
            unset SANDBOX_PID
        fi
        echo "[INFO] Exporting $CONTAINER_NAME -> $EXPORT_FILE"
        enroot export -o "$EXPORT_FILE" "$CONTAINER_NAME"
        echo "[INFO] Exported. Use with run_sandbox_retry.sh:"
        echo "  bash scripts/sandbox/run_sandbox_retry.sh -i $EXPORT_FILE"
    fi
fi

echo ""
echo "[INFO] Setup complete."
