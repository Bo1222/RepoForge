# T7 official Harness validation tools

These tools are deliberately **outside** the frozen Agent source
(`repoforge/`, `scripts/`). The official Harness exporters and reporters are
read-only with respect to an experiment candidate. The preparation tools below
write only their explicitly named candidate audit files.

## Why this directory exists

The T7 local evaluation already uses the official SWE-bench Verified
`test_patch`, `FAIL_TO_PASS`/`PASS_TO_PASS`, the official
`make_test_spec` eval script and the official `get_logs_eval` /
`get_eval_report` parsers. It does **not** run the official Harness container
or its official images. The official Harness result must therefore be reported
separately as `official_resolved`; it is not the same as `local_resolved`.

## Frozen ID sets

`export_official.py --phase ids` writes `id-manifest.json` and
`hash-manifest.json` with stable, hashed lists:

- dev: 30 instance IDs, 5 arms × 30 = 150 jobs;
- holdout frozen: 100 instance IDs, 5 arms × 100 = 500 job definitions;
- holdout execution: the locked two arms only (A3 + A0), 2 × 100 = 200 jobs.

Every arm must match the frozen split exactly. The manifest records
`dev_sha256`, `holdout_sha256`, per-arm job hashes, `manifest_hash`,
`lock_hash`, dataset fingerprint (when the cache is available) and the
analysis-plan hash. This prevents task/answer mismatches when the official
Harness later matches `instance_id` values.

## Other T7 preparation tools

- `prepare_clean_indexes.py` builds or reuses shared indexes only for the
  scan-clean instances and writes `holdout-index-coverage.json`
  (`clean_complete=true` only when every clean instance is ready). Non-clean
  instances are listed in `failed` and are never indexed or executed.
- `rebuild_smoke_envs.py` replays the frozen smoke `recipe.json` at the
  original path strings, restores the original evidence files and `ready.json`,
  and verifies each `native-gate-provision.json` receipt byte for byte. It does
  not rewrite the gate snapshot. `conda list --json` is run exactly as in
  `NativeProvisioner.prepare`, which also drops stale conda-meta records for
  packages replaced by pip.
- `holdout_progress.py` is a read-only, compact holdout screen for
  `watch -n 15 -d`: clean-job counts, per-arm resolved, current job/stage,
  recent job durations/costs, ETA, budget guard, skipped non-clean arms and
  free disk. It never writes to the candidate.

## Usage

Run from the repository root.

```bash
# 1. Fix ID lists and per-job job IDs (safe to run before patches exist)
.venv/bin/python analysis-tools/export_official.py \
  --candidate experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z \
  --phase ids

# 2. After dev is complete: write dev predictions and patch/answer hashes
.venv/bin/python analysis-tools/export_official.py \
  --candidate experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z \
  --phase dev

# 3. After the locked holdout run: write holdout predictions
.venv/bin/python analysis-tools/export_official.py \
  --candidate experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z \
  --phase holdout

# 4. Everything (dev + holdout)
.venv/bin/python analysis-tools/export_official.py \
  --candidate experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z \
  --phase all
```

Outputs (default):

```text
analysis-tools/artifacts/<candidate>/id-manifest.json
analysis-tools/artifacts/<candidate>/hash-manifest.json
analysis-tools/artifacts/<candidate>/predictions/dev-a0.jsonl ... dev-a4.jsonl
analysis-tools/artifacts/<candidate>/predictions/holdout-a3.jsonl
analysis-tools/artifacts/<candidate>/predictions/holdout-a0.jsonl
```

Each predictions file contains only the official fields:

```json
{"instance_id": "...", "model_name_or_path": "repoforge-t7-dev-a3", "model_patch": "..."}
```

One file per phase/arm is used so the same `instance_id` can appear in
different arms without duplicate `(instance_id)` entries in one file.

## Run the official Harness

The official command in the installed `swebench` version is:

```bash
.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path analysis-tools/artifacts/<candidate>/predictions/dev-a3.jsonl \
  --max_workers 4 \
  --timeout 1800 \
  --run_id t7-dev-a3
```

Use the official evaluation environment (Docker images / official runner) and
one `--run_id` per arm. The Harness writes per-instance
`logs/run_evaluation/<run_id>/<model>/<instance_id>/report.json`.

`official_harness_template.sh` is a thin template for this command.

## Compare official vs local

```bash
.venv/bin/python analysis-tools/report_official.py \
  --candidate experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z \
  --official logs/run_evaluation/t7-dev-a3 \
  --phase dev
```

The tool accepts:

- a directory containing official per-instance `report.json` files;
- an official aggregate report with `resolved_ids`/`unresolved_ids`;
- a JSONL file with `instance_id`, `model_name_or_path`, `resolved`.

Outputs:

```text
analysis-tools/artifacts/<candidate>/report/official-comparison.json
analysis-tools/artifacts/<candidate>/report/official-rows.jsonl
analysis-tools/artifacts/<candidate>/report/official-report.md
```

The report shows per-arm local resolved, official resolved, local-only,
official-only, both, official-not-run, and unmatched official IDs.

## Rules

- Official results never change `t7-lock.json`, the winner, task IDs,
  prompts, tools, budgets or `local_resolved`.
- Empty patches / environment-blocked instances stay in the denominator and
  are reported as missing/empty, never as resolved.
- Official Harness running is an independent validation item. Until it
  actually completes, keep `official_harness=not_run` and
  `official_resolved=null`.
- Do not add these files to `repoforge/` or `scripts/`; doing so would change
  the frozen source identity.
