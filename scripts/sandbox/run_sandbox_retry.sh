#!/usr/bin/env bash
set -u
set -o pipefail

# -------- defaults (override via flags or env) --------
# The default image should be pre-configured with all packages from
# sandbox-requirements.txt. Create it with:
#   bash scripts/sandbox/setup_sandbox.sh --export sr2am_sandbox.sqsh
ENROOT_NAME="${ENROOT_NAME:-sandbox-server}"
ENROOT_IMAGE="${ENROOT_IMAGE:-sr2am_sandbox.sqsh}"

RETRIES="${RETRIES:-100}"          # total attempts
DELAY_SEC="${DELAY_SEC:-3}"       # delay between attempts
BACKOFF="${BACKOFF:-0}"           # if 1, delay grows each attempt (linear)
KEEP_ON_FINAL_FAIL="${KEEP_ON_FINAL_FAIL:-0}"  # if 1, don't remove container after last failure
LOG_DIR="${LOG_DIR:-./sandbox_logs}"

CMD="${CMD:-make run-online}"     # what to run inside the container

# -------- helpers --------
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -n NAME        Enroot container name (default: $ENROOT_NAME)
  -i IMAGE       Enroot image/sqsh (default: $ENROOT_IMAGE)
  -r RETRIES     Number of attempts (default: $RETRIES)
  -d SECONDS     Base delay between attempts (default: $DELAY_SEC)
  -b             Enable linear backoff (delay * attempt)
  -k             Keep container on final failure (for debugging)
  -c CMD         Command to run inside container (default: "$CMD")
  -l LOG_DIR     Log directory (default: $LOG_DIR)
  -h             Help

Env vars also work: ENROOT_NAME, ENROOT_IMAGE, RETRIES, DELAY_SEC, BACKOFF, KEEP_ON_FINAL_FAIL, CMD, LOG_DIR
EOF
}

have() { command -v "$1" >/dev/null 2>&1; }

container_exists() {
  # enroot list prints names; match exact name on its own line
  enroot list 2>/dev/null | awk '{print $1}' | grep -Fxq "$ENROOT_NAME"
}

remove_container() {
  if container_exists; then
    echo "[enroot] removing existing container: $ENROOT_NAME"
    enroot remove -f "$ENROOT_NAME" || true
  fi
}

create_container() {
  echo "[enroot] creating container: $ENROOT_NAME from image: $ENROOT_IMAGE"
  enroot create --name "$ENROOT_NAME" "$ENROOT_IMAGE"
}

run_container() {
  local attempt="$1"
  local log_file="$2"

  echo "[enroot] starting container (attempt $attempt): $ENROOT_NAME"
  # Capture stdout+stderr so you can inspect failures later
  enroot start --rw "$ENROOT_NAME" bash -lc "$CMD" 2>&1 | tee "$log_file"
  return "${PIPESTATUS[0]}"  # exit code of enroot start (not tee)
}

sleep_between_attempts() {
  local attempt="$1"
  local sleep_for="$DELAY_SEC"
  if [[ "$BACKOFF" == "1" ]]; then
    sleep_for=$((DELAY_SEC * attempt))
  fi
  echo "[retry] sleeping ${sleep_for}s before next attempt..."
  sleep "$sleep_for"
}

# -------- flags --------
while getopts ":n:i:r:d:bkc:l:h" opt; do
  case "$opt" in
    n) ENROOT_NAME="$OPTARG" ;;
    i) ENROOT_IMAGE="$OPTARG" ;;
    r) RETRIES="$OPTARG" ;;
    d) DELAY_SEC="$OPTARG" ;;
    b) BACKOFF="1" ;;
    k) KEEP_ON_FINAL_FAIL="1" ;;
    c) CMD="$OPTARG" ;;
    l) LOG_DIR="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Missing arg for -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

# -------- sanity checks --------
if ! have enroot; then
  echo "ERROR: enroot not found in PATH" >&2
  exit 127
fi

mkdir -p "$LOG_DIR"

node_name="$(hostname -s 2>/dev/null || hostname)"
echo "[info] node: $node_name"
echo "[info] name: $ENROOT_NAME"
echo "[info] image: $ENROOT_IMAGE"
echo "[info] retries: $RETRIES, delay: $DELAY_SEC, backoff: $BACKOFF"
echo "[info] cmd: $CMD"
echo "[info] logs: $LOG_DIR"

# -------- main loop --------
for attempt in $(seq 1 "$RETRIES"); do
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="${LOG_DIR}/${ENROOT_NAME}_attempt${attempt}_${ts}.log"

  echo "============================================================"
  echo "[attempt $attempt/$RETRIES] $(date)"
  echo "============================================================"

  # Always remove+recreate (matches what you said you do on repeats)
  remove_container

  if ! create_container 2>&1 | tee "$log_file.create"; then
    echo "[attempt $attempt] create failed (see $log_file.create)"
    if [[ "$attempt" -lt "$RETRIES" ]]; then
      sleep_between_attempts "$attempt"
      continue
    else
      exit 1
    fi
  fi

  if run_container "$attempt" "$log_file"; then
    echo "[success] sandbox started successfully on attempt $attempt"
    exit 0
  else
    rc=$?
    echo "[fail] sandbox exited with code $rc (see $log_file)"
    if [[ "$attempt" -lt "$RETRIES" ]]; then
      sleep_between_attempts "$attempt"
      continue
    fi

    echo "[final fail] all attempts exhausted."
    if [[ "$KEEP_ON_FINAL_FAIL" != "1" ]]; then
      echo "[cleanup] removing container after final failure"
      remove_container
    else
      echo "[debug] keeping container '$ENROOT_NAME' for inspection"
    fi
    exit "$rc"
  fi
done
