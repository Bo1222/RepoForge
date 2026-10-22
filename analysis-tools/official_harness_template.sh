#!/usr/bin/env bash
# Template for one T7 official Harness run.  Run this in the official
# SWE-bench evaluation environment (Docker images / official runner), not in
# the frozen Agent source tree.
#
# Usage:
#   ./analysis-tools/official_harness_template.sh <predictions.jsonl> <run_id> [workers]
set -euo pipefail

PREDICTIONS="${1:?usage: official_harness_template.sh <predictions.jsonl> <run_id> [workers]}"
RUN_ID="${2:?usage: official_harness_template.sh <predictions.jsonl> <run_id> [workers]}"
WORKERS="${3:-4}"

python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path "${PREDICTIONS}" \
  --max_workers "${WORKERS}" \
  --timeout 1800 \
  --run_id "${RUN_ID}"
