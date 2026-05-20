#!/bin/bash
# =============================================================================
# run_inference_slurm_8b.sh -- SLURM inference for 8B model (1 node, 8 GPUs)
#
# Launches SGLang with TP=1, DP=8, routes requests through sglang_router,
# then runs run_agent.py.
#
# Usage:
#   sbatch scripts/run_inference_slurm_8b.sh \
#     ~/models/SR2AM-v0.1-8B \
#     SR2AM-v0.1-8B \
#     ~/data/test_questions.jsonl \
#     sr2am-v0.1-8b-results.jsonl \
#     Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
#     http://SUMMARIZER_HOST:30000/v1 \
#     64 1 \
#     "--fix_datetime --filtering --remove_tags --no_break_early --agent_type think --max_turns 50 --max_completion_tokens 16384 --temperature 0.8 --code_sandbox_servers SANDBOX_HOST1 SANDBOX_HOST2"
# =============================================================================

#SBATCH --job-name=sr2am-infer-8b
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --account=research
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --time=1000-00:00:00
#SBATCH --output=logs/sr2am-infer-8b-%j.out

set -x

# ========================= Parse Positional Arguments ========================
model_path=$1
model_name=$2
input_file_name=$3
output_file_name=$4
browsing_summarize_model=$5
browsing_summarize_model_base_url=$6
max_concurrent=$7
num_retries=$8
additional_args=$9

if [ -z "$model_path" ] || [ -z "$model_name" ] || [ -z "$input_file_name" ] || \
   [ -z "$output_file_name" ] || [ -z "$browsing_summarize_model" ] || [ -z "$browsing_summarize_model_base_url" ] || \
   [ -z "$max_concurrent" ] || [ -z "$num_retries" ] || [ -z "$additional_args" ]; then
    cat <<'EOF'
Usage: sbatch scripts/run_inference_slurm_8b.sh \
  MODEL_PATH MODEL_NAME INPUT_FILE OUTPUT_FILE \
  BROWSING_MODEL BROWSING_URL MAX_CONCURRENT NUM_RETRIES "ADDITIONAL_ARGS"
EOF
    exit 1
fi

# ========================= Change to repo root and activate venv =============
# SLURM copies scripts to /var/spool, so use SLURM_SUBMIT_DIR instead of BASH_SOURCE
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
echo "[INFO] Working directory: $(pwd)"

if [[ -f .venv/bin/activate ]]; then
    echo "[INFO] Activating .venv"
    source .venv/bin/activate
fi

# ========================= SGLang Config =====================================
MODEL_PATH=$model_path
MODEL_NAME=$model_name
CTX_LEN=40960
TP_PER_INSTANCE=1
DP_PER_NODE=8
WORKER_PORT=8000
ROUTER_PORT=30000
ROUTER_POLICY="cache_aware"

mkdir -p logs
SGLANG_LOG_DIR="$(pwd)/sglang_logs"
mkdir -p "$SGLANG_LOG_DIR"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
HEAD_NODE="${NODES[0]}"

echo "[INFO] Nodes: ${NODES[*]}"
echo "[INFO] Head node (router): $HEAD_NODE"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CPU_ALLOC_CONF=expandable_segments:True

# ========================= Start SGLang Workers ==============================
for NODE in "${NODES[@]}"; do
  srun -N1 -n1 -w "$NODE" --gres=gpu:8 \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --cpu-bind=cores --exclusive \
    bash -lc "python -m sglang.launch_server \
      --model-path '${MODEL_PATH}' \
      --served-model-name '${MODEL_NAME}' \
      --context-length '${CTX_LEN}' \
      --tp '${TP_PER_INSTANCE}' \
      --dp '${DP_PER_NODE}' \
      --tool-call-parser qwen \
      --host 0.0.0.0 \
      --port '${WORKER_PORT}' \
      >> ${SGLANG_LOG_DIR}/sglang_worker_${SLURM_JOB_ID}_${NODE}_${MODEL_NAME}.log 2>&1" &
done

sleep 180

# ========================= Start Router ======================================
WORKER_URLS=()
for NODE in "${NODES[@]}"; do
  WORKER_URLS+=("http://${NODE}:${WORKER_PORT}")
done
printf "[INFO] Router will point to workers:\n"
for URL in "${WORKER_URLS[@]}"; do
  printf "  - %s\n" "$URL"
done

if [[ "$HOSTNAME" == "$HEAD_NODE" ]]; then
    python -m sglang_router.launch_router \
    --worker-urls ${WORKER_URLS[*]} \
    --host 0.0.0.0 \
    --port ${ROUTER_PORT} \
    --policy ${ROUTER_POLICY} \
    >> ${SGLANG_LOG_DIR}/sglang_router_${SLURM_JOB_ID}_${MODEL_NAME}.log 2>&1 &
fi

sleep 60

# ========================= Run Inference =====================================
source .env

python run_agent.py \
    --input_file "$input_file_name" \
    --model "$model_name" \
    --model_base_url "http://$HEAD_NODE:$ROUTER_PORT/v1" \
    --model_timeout 300 \
    --output_file "outputs/$output_file_name" \
    --browsing_summarize_model "$browsing_summarize_model" \
    --browsing_summarize_model_base_url "$browsing_summarize_model_base_url" \
    --max_concurrent "$max_concurrent" \
    --num_retries "$num_retries" \
    $additional_args
