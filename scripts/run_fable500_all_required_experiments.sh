#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET_DIR="."
INPUT="$DATASET_DIR/benchmark/inputs/test100_cases.jsonl"
MASKED_INPUT="$DATASET_DIR/benchmark/inputs/test100_cases_masked_report.jsonl"
OUT_DIR="$DATASET_DIR/benchmark/outputs"
ANALYSIS_DIR="$DATASET_DIR/benchmark/analysis"
LOG_DIR="$DATASET_DIR/benchmark/logs"

PROXY_URL="${FABLE500_PROXY_URL:-}"
MAX_TOKENS="${FABLE500_MAX_TOKENS:-1600}"
FULLCASE_WORKERS="${FABLE500_FULLCASE_WORKERS:-3}"
GPT_AUX_WORKERS="${FABLE500_GPT_AUX_WORKERS:-1}"
WORKFLOW_WORKERS="${FABLE500_WORKFLOW_WORKERS:-1}"
TIMEOUT="${FABLE500_TIMEOUT:-240}"
RETRIES="${FABLE500_RETRIES:-2}"
WORKFLOW_MAX_TOKENS_MODULE="${FABLE500_WORKFLOW_MAX_TOKENS_MODULE:-1600}"
WORKFLOW_MAX_TOKENS_FINAL="${FABLE500_WORKFLOW_MAX_TOKENS_FINAL:-1600}"
BOOTSTRAP="${FABLE500_BOOTSTRAP:-2000}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -n "$PROXY_URL" ]]; then
  export FABLE500_PROXY_URL="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
fi
export FABLE500_BAILIAN_VISION_MODEL="${FABLE500_BAILIAN_VISION_MODEL:-qwen3.7-plus}"

FORCE_FLAG=()
if [[ "${FABLE500_FORCE:-0}" == "1" ]]; then
  FORCE_FLAG=(--force)
fi

mkdir -p "$OUT_DIR" "$ANALYSIS_DIR" "$LOG_DIR"

echo "== FABLE-500 required experiment suite =="
echo "proxy=$PROXY_URL"
echo "max_tokens=$MAX_TOKENS fullcase_workers=$FULLCASE_WORKERS gpt_aux_workers=$GPT_AUX_WORKERS workflow_workers=$WORKFLOW_WORKERS"
echo "force=${FABLE500_FORCE:-0}"
echo

echo "== Step 0: export benchmark inputs =="
python "$DATASET_DIR/scripts/export_benchmark_inputs.py"
python "$DATASET_DIR/scripts/export_masked_report_benchmark_inputs.py"

echo
echo "== Step 1: API environment check =="
python "$DATASET_DIR/scripts/check_api_environment.py" \
  --providers openrouter_gpt56_sol,openrouter_gemini37_flash,openrouter_claude_sonnet5,bailian_qwen_vl \
  --timeout 20

run_api_benchmark() {
  local label="$1"
  local provider="$2"
  local systems="$3"
  local input_file="$4"
  local output_file="$5"
  local workers="$6"
  local log_file="$7"

  echo "== Launch $label =="
  python "$DATASET_DIR/scripts/run_api_benchmark.py" \
    --input "$input_file" \
    --output "$output_file" \
    --systems "$systems" \
    --provider-chain "$provider" \
    --proxy-url "$PROXY_URL" \
    --max-tokens "$MAX_TOKENS" \
    --workers "$workers" --timeout "$TIMEOUT" --retries "$RETRIES" "${FORCE_FLAG[@]}" \
    2>&1 | tee "$log_file"
}

run_workflow() {
  echo "== Launch GPT-5.6 Sol four-stage workflow comparator =="
  python "$DATASET_DIR/scripts/run_api_workflow_benchmark.py" \
    --input "$INPUT" \
    --output "$OUT_DIR/api_workflow_openrouter_gpt56sol.jsonl" \
    --provider-chain openrouter_gpt56_sol \
    --proxy-url "$PROXY_URL" \
    --max-tokens-module "$WORKFLOW_MAX_TOKENS_MODULE" \
    --max-tokens-final "$WORKFLOW_MAX_TOKENS_FINAL" \
    --workers "$WORKFLOW_WORKERS" --timeout "$TIMEOUT" --retries "$RETRIES" "${FORCE_FLAG[@]}" \
    2>&1 | tee "$LOG_DIR/api_workflow_openrouter_gpt56sol.log"
}

wait_group() {
  local failed=0
  local pid
  for pid in "$@"; do
    if wait "$pid"; then
      echo "process $pid completed"
    else
      echo "process $pid failed" >&2
      failed=1
    fi
  done
  return "$failed"
}

echo
echo "== Step 2: full-case model panel, parallel =="
run_api_benchmark \
  "GPT-5.6 Sol full-case" \
  openrouter_gpt56_sol \
  full_case_multimodal \
  "$INPUT" \
  "$OUT_DIR/test100_fullcase_openrouter_gpt56sol.jsonl" \
  "$FULLCASE_WORKERS" \
  "$LOG_DIR/test100_fullcase_openrouter_gpt56sol.log" &
pid_gpt=$!

run_api_benchmark \
  "Gemini 3.7 Flash full-case" \
  openrouter_gemini37_flash \
  full_case_multimodal \
  "$INPUT" \
  "$OUT_DIR/test100_fullcase_openrouter_gemini37flash.jsonl" \
  "$FULLCASE_WORKERS" \
  "$LOG_DIR/test100_fullcase_openrouter_gemini37flash.log" &
pid_gemini=$!

run_api_benchmark \
  "Claude Sonnet 5 full-case" \
  openrouter_claude_sonnet5 \
  full_case_multimodal \
  "$INPUT" \
  "$OUT_DIR/test100_fullcase_openrouter_claude_sonnet5.jsonl" \
  "$FULLCASE_WORKERS" \
  "$LOG_DIR/test100_fullcase_openrouter_claude_sonnet5.log" &
pid_claude=$!

run_api_benchmark \
  "Qwen3.7 Plus full-case" \
  bailian_qwen_vl \
  full_case_multimodal \
  "$INPUT" \
  "$OUT_DIR/test100_fullcase_bailian_qwen37plus.jsonl" \
  "$FULLCASE_WORKERS" \
  "$LOG_DIR/test100_fullcase_bailian_qwen37plus.log" &
pid_qwen=$!

wait_group "$pid_gpt" "$pid_gemini" "$pid_claude" "$pid_qwen"

echo
echo "== Step 3: GPT-5.6 Sol auxiliary experiments, parallel =="
run_api_benchmark \
  "GPT-5.6 Sol input ablation" \
  openrouter_gpt56_sol \
  all \
  "$INPUT" \
  "$OUT_DIR/test100_ablation_openrouter_gpt56sol.jsonl" \
  "$GPT_AUX_WORKERS" \
  "$LOG_DIR/test100_ablation_openrouter_gpt56sol.log" &
pid_ablation=$!

run_api_benchmark \
  "GPT-5.6 Sol masked-report sensitivity" \
  openrouter_gpt56_sol \
  report_text_only,full_case_multimodal \
  "$MASKED_INPUT" \
  "$OUT_DIR/test100_masked_report_openrouter_gpt56sol.jsonl" \
  "$GPT_AUX_WORKERS" \
  "$LOG_DIR/test100_masked_report_openrouter_gpt56sol.log" &
pid_masked=$!

run_workflow &
pid_workflow=$!

wait_group "$pid_ablation" "$pid_masked" "$pid_workflow"

echo
echo "== Step 4: analysis =="
python "$DATASET_DIR/scripts/analyze_multimodel_fullcase.py" \
  --predictions \
    "$OUT_DIR/test100_fullcase_openrouter_gpt56sol.jsonl" \
    "$OUT_DIR/test100_fullcase_openrouter_gemini37flash.jsonl" \
    "$OUT_DIR/test100_fullcase_openrouter_claude_sonnet5.jsonl" \
    "$OUT_DIR/test100_fullcase_bailian_qwen37plus.jsonl" \
  --output-dir "$ANALYSIS_DIR/model_family_fullcase_test100"

python "$DATASET_DIR/scripts/analyze_api_reference_benchmark.py" \
  --single-call-predictions "$OUT_DIR/test100_ablation_openrouter_gpt56sol.jsonl" \
  --workflow-predictions "$OUT_DIR/api_workflow_openrouter_gpt56sol.jsonl" \
  --output-dir "$ANALYSIS_DIR/api_reference_test100" \
  --bootstrap "$BOOTSTRAP"

python "$DATASET_DIR/scripts/analyze_api_benchmark.py" \
  --predictions "$OUT_DIR/test100_masked_report_openrouter_gpt56sol.jsonl" \
  --output-dir "$ANALYSIS_DIR/masked_report_openrouter_gpt56sol" \
  --bootstrap "$BOOTSTRAP"

echo
echo "== Complete. Key outputs =="
echo "$ANALYSIS_DIR/model_family_fullcase_test100/current_model_fullcase_results.xlsx"
echo "$ANALYSIS_DIR/api_reference_test100/api_reference_benchmark_results.xlsx"
echo "$ANALYSIS_DIR/masked_report_openrouter_gpt56sol/api_benchmark_results.xlsx"
