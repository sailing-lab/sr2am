#!/bin/bash
# =============================================================================
# run_inference_local.sh -- Single-machine SR2AM inference (no SLURM required)
#
# Starts an SGLang server, runs the SR2AM agent on an input dataset, and
# optionally evaluates the results.
#
# Requirements: 4-8 GPUs (Hopper recommended), SGLang, Python 3.10+
# =============================================================================

set -euo pipefail

# ========================= Default Configuration =============================
MODEL_PATH=""
MODEL_NAME=""
MODEL_SIZE=""          # "8b" or "30b"
INPUT_FILE=""
OUTPUT_FILE=""
NUM_GPUS=8
AGENT_TYPE="think"
MAX_TURNS=""               # default: 50 for 8b, 100 for 30b
MAX_COMPLETION_TOKENS=16384
TEMPERATURE=""         # default: 0.8 for 8b, 1.0 for 30b
MAX_CONCURRENT=64
BROWSING_SUMMARIZE_MODEL=""
BROWSING_SUMMARIZE_URL=""
CODE_SANDBOX_SERVERS=""
EXTRA_ARGS=""
EVALUATE=false
SGLANG_PORT=8000
SGLANG_CONTEXT_LENGTH=""  # default: auto based on model size
# =============================================================================

usage() {
    cat <<'EOF'
Usage: bash scripts/run_inference_local.sh [OPTIONS]

Required:
  --model-path PATH          Path to HuggingFace model directory
  --model-name NAME          Served model name (e.g., SR2AM-v1.0-30B)
  --model-size SIZE          Model size: "8b" or "30b"
  --input-file FILE          Input JSONL file with questions
  --output-file FILE         Output JSONL file for results

Optional:
  --num-gpus N               Number of GPUs (default: 8)
  --agent-type TYPE          Agent type: think|configurator|instruct (default: think)
  --max-turns N              Max agent turns (default: 50 for 8b, 100 for 30b)
  --max-completion-tokens N  Max tokens per turn (default: 16384)
  --temperature T            Sampling temperature (default: 0.8 for 8b, 1.0 for 30b)
  --max-concurrent N         Max concurrent requests (default: 64)
  --browsing-summarize-model MODEL  Browsing summarizer model name
  --browsing-summarize-url URL      Browsing summarizer endpoint
  --code-sandbox-servers "H1 H2"   Space-separated sandbox hostnames (quoted)
  --sglang-port PORT         SGLang server port (default: 8000)
  --context-length N         Max context length (default: auto)
  --extra-args "ARGS"        Additional arguments passed to run_agent.py (quoted)
  --evaluate                 Run evaluation after inference
  --help                     Show this help message

Note: --code-sandbox-servers and --extra-args take a single quoted string.
  The string is word-split when passed to run_agent.py.

  Paper reproduction: pass --fix_datetime via --extra-args to use the fixed
  training datetime. SerpAPI is the default search provider.

Examples:
  # 8B model on 8 GPUs (TP=1, DP=8, temp=0.8, max_turns=50)
  bash scripts/run_inference_local.sh \
    --model-path ./models/SR2AM-v0.1-8B \
    --model-name SR2AM-v0.1-8B \
    --model-size 8b \
    --input-file data/test_questions.jsonl \
    --output-file outputs/results.jsonl \
    --browsing-summarize-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --browsing-summarize-url http://SUMMARIZER_HOST:30000/v1 \
    --code-sandbox-servers "SANDBOX_HOST1 SANDBOX_HOST2" \
    --extra-args "--fix_datetime" \
    --evaluate

  # 30B model on 8 GPUs (TP=4, DP=2, temp=1.0, max_turns=100)
  bash scripts/run_inference_local.sh \
    --model-path ./models/SR2AM-v1.0-30B \
    --model-name SR2AM-v1.0-30B \
    --model-size 30b \
    --input-file data/test_questions.jsonl \
    --output-file outputs/results.jsonl \
    --browsing-summarize-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --browsing-summarize-url http://SUMMARIZER_HOST:30000/v1 \
    --code-sandbox-servers "SANDBOX_HOST1 SANDBOX_HOST2" \
    --extra-args "--fix_datetime --code_concurrency 128" \
    --evaluate

  # 30B model on 4 GPUs (TP=4, DP=1)
  bash scripts/run_inference_local.sh \
    --model-path ./models/SR2AM-v1.0-30B \
    --model-name SR2AM-v1.0-30B \
    --model-size 30b \
    --num-gpus 4 \
    --input-file data/test_questions.jsonl \
    --output-file outputs/results.jsonl \
    --browsing-summarize-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --browsing-summarize-url http://SUMMARIZER_HOST:30000/v1 \
    --code-sandbox-servers "SANDBOX_HOST1 SANDBOX_HOST2" \
    --extra-args "--fix_datetime"
EOF
    exit 0
}

# ========================= Parse Arguments ===================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)       MODEL_PATH="$2"; shift 2 ;;
        --model-name)       MODEL_NAME="$2"; shift 2 ;;
        --model-size)       MODEL_SIZE="$2"; shift 2 ;;
        --input-file)       INPUT_FILE="$2"; shift 2 ;;
        --output-file)      OUTPUT_FILE="$2"; shift 2 ;;
        --num-gpus)         NUM_GPUS="$2"; shift 2 ;;
        --agent-type)       AGENT_TYPE="$2"; shift 2 ;;
        --max-turns)        MAX_TURNS="$2"; shift 2 ;;
        --max-completion-tokens) MAX_COMPLETION_TOKENS="$2"; shift 2 ;;
        --temperature)      TEMPERATURE="$2"; shift 2 ;;
        --max-concurrent)   MAX_CONCURRENT="$2"; shift 2 ;;
        --browsing-summarize-model) BROWSING_SUMMARIZE_MODEL="$2"; shift 2 ;;
        --browsing-summarize-url)   BROWSING_SUMMARIZE_URL="$2"; shift 2 ;;
        --code-sandbox-servers)     CODE_SANDBOX_SERVERS="$2"; shift 2 ;;
        --sglang-port)      SGLANG_PORT="$2"; shift 2 ;;
        --context-length)   SGLANG_CONTEXT_LENGTH="$2"; shift 2 ;;
        --extra-args)       EXTRA_ARGS="$2"; shift 2 ;;
        --evaluate)         EVALUATE=true; shift ;;
        --help)             usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ========================= Validate Required Args ============================
missing=()
[[ -z "$MODEL_PATH" ]] && missing+=("--model-path")
[[ -z "$MODEL_NAME" ]] && missing+=("--model-name")
[[ -z "$MODEL_SIZE" ]] && missing+=("--model-size")
[[ -z "$INPUT_FILE" ]] && missing+=("--input-file")
[[ -z "$OUTPUT_FILE" ]] && missing+=("--output-file")

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Error: Missing required arguments: ${missing[*]}"
    echo "Run with --help for usage information."
    exit 1
fi

if [[ "$MODEL_SIZE" != "8b" && "$MODEL_SIZE" != "30b" ]]; then
    echo "Error: --model-size must be '8b' or '30b', got '$MODEL_SIZE'"
    exit 1
fi

# ========================= Compute TP/DP =====================================
if [[ "$MODEL_SIZE" == "8b" ]]; then
    TP=1
    DP=$NUM_GPUS
    [[ -z "$TEMPERATURE" ]] && TEMPERATURE=0.8
    [[ -z "$MAX_TURNS" ]] && MAX_TURNS=50
    [[ -z "$SGLANG_CONTEXT_LENGTH" ]] && SGLANG_CONTEXT_LENGTH=40960
else
    TP=4
    DP=$((NUM_GPUS / 4))
    [[ -z "$TEMPERATURE" ]] && TEMPERATURE=1.0
    [[ -z "$MAX_TURNS" ]] && MAX_TURNS=100
    [[ -z "$SGLANG_CONTEXT_LENGTH" ]] && SGLANG_CONTEXT_LENGTH=131072
    if [[ $DP -lt 1 ]]; then
        echo "Error: 30B model requires at least 4 GPUs (TP=4). Got --num-gpus $NUM_GPUS"
        exit 1
    fi
fi

echo "============================================================"
echo "SR2AM Inference Configuration"
echo "============================================================"
echo "  Model:       $MODEL_NAME ($MODEL_SIZE)"
echo "  Model path:  $MODEL_PATH"
echo "  Input:       $INPUT_FILE"
echo "  Output:      $OUTPUT_FILE"
echo "  GPUs:        $NUM_GPUS (TP=$TP, DP=$DP)"
echo "  Agent type:  $AGENT_TYPE"
echo "  Temperature: $TEMPERATURE"
echo "  Context len: $SGLANG_CONTEXT_LENGTH"
echo "  Evaluate:    $EVALUATE"
echo "============================================================"

# ========================= Change to repo root ===============================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ========================= Source Environment ================================
if [[ -f .env ]]; then
    echo "[INFO] Loading .env file"
    set -a
    source .env
    set +a
fi

# ========================= Cleanup on exit ===================================
SGLANG_PID=""
cleanup() {
    if [[ -n "$SGLANG_PID" ]]; then
        echo "[INFO] Stopping SGLang server (PID: $SGLANG_PID) and all child processes..."
        # Kill the entire process group (SGLang spawns DP controllers, detokenizers, TP workers)
        pkill -P "$SGLANG_PID" 2>/dev/null || true
        kill "$SGLANG_PID" 2>/dev/null || true
        wait "$SGLANG_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ========================= Start SGLang Server ===============================
mkdir -p logs

echo "[INFO] Starting SGLang server on port $SGLANG_PORT..."
SGLANG_CMD="python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --context-length $SGLANG_CONTEXT_LENGTH \
    --tp $TP \
    --dp $DP \
    --tool-call-parser qwen \
    --host 0.0.0.0 \
    --port $SGLANG_PORT"

echo "[INFO] Command: $SGLANG_CMD"
$SGLANG_CMD > logs/sglang_server.log 2>&1 &
SGLANG_PID=$!

# ========================= Wait for Server Ready =============================
echo "[INFO] Waiting for SGLang server to be ready..."
MAX_WAIT=600
WAIT_INTERVAL=10
ELAPSED=0

while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    if curl -s "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
        echo "[INFO] SGLang server is ready (took ${ELAPSED}s)"
        break
    fi
    # Check if process is still running
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
        echo "[ERROR] SGLang server process died. Check logs/sglang_server.log"
        exit 1
    fi
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    echo "[ERROR] SGLang server did not become ready within ${MAX_WAIT}s"
    echo "[ERROR] Check logs/sglang_server.log for details"
    exit 1
fi

# ========================= Build run_agent.py Arguments ======================
MODEL_BASE_URL="http://localhost:${SGLANG_PORT}/v1"

AGENT_ARGS=(
    --input_file "$INPUT_FILE"
    --model "$MODEL_NAME"
    --model_base_url "$MODEL_BASE_URL"
    --model_timeout 300
    --output_file "$OUTPUT_FILE"
    --max_concurrent "$MAX_CONCURRENT"
    --num_retries 1
    --agent_type "$AGENT_TYPE"
    --max_turns "$MAX_TURNS"
    --max_completion_tokens "$MAX_COMPLETION_TOKENS"
    --temperature "$TEMPERATURE"
    --filtering
    --remove_tags
    --no_break_early
)

if [[ -n "$BROWSING_SUMMARIZE_MODEL" ]]; then
    AGENT_ARGS+=(--browsing_summarize_model "$BROWSING_SUMMARIZE_MODEL")
fi
if [[ -n "$BROWSING_SUMMARIZE_URL" ]]; then
    AGENT_ARGS+=(--browsing_summarize_model_base_url "$BROWSING_SUMMARIZE_URL")
fi
if [[ -n "$CODE_SANDBOX_SERVERS" ]]; then
    AGENT_ARGS+=(--code_sandbox_servers $CODE_SANDBOX_SERVERS)
fi

# ========================= Run Inference =====================================
echo "[INFO] Running inference..."
mkdir -p "$(dirname "$OUTPUT_FILE")"

python run_agent.py "${AGENT_ARGS[@]}" $EXTRA_ARGS

echo "[INFO] Inference complete. Output: $OUTPUT_FILE"

# ========================= Optional Evaluation ===============================
if [[ "$EVALUATE" == true ]]; then
    echo "[INFO] Running evaluation..."
    python evaluation/compute_rep_results.py \
        --input_file "$OUTPUT_FILE" \
        --num_reps 1
    echo "[INFO] Evaluation complete."
fi

echo "[INFO] Done."
