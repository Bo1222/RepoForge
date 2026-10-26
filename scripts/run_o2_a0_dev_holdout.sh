#!/usr/bin/env bash
# Foreground runner; use nohup to detach. No status/result polling or retries.
set -euo pipefail

PAIRED_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PAIRED_PYTHON="$PAIRED_PROJECT_ROOT/.venv/bin/python"

if [[ $# -ne 1 || "$1" == "--help" || "$1" == "-h" ]]; then
  printf 'Usage: bash scripts/run_o2_a0_dev_holdout.sh /absolute/path/to/prepared-candidate\n'
  printf 'Runs preflight -> smoke -> dev -> holdout, stopping on the first error.\n'
  [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]] && exit 0
  exit 2
fi

PAIRED_RUN="$(cd -- "$1" && pwd)"
[[ -f "$PAIRED_RUN/manifest.json" && -f "$PAIRED_RUN/jobs.json" ]] || {
  printf 'Missing prepared manifest.json/jobs.json: %s\n' "$PAIRED_RUN" >&2
  exit 2
}
cd -- "$PAIRED_PROJECT_ROOT"

# Keep one launcher for the whole pipeline. The Python scheduler independently
# locks each phase, validates frozen evidence and skips already frozen phases.
exec 9>"$PAIRED_RUN/dev-holdout-runner.lock"
flock -n 9 || { printf 'A dev/holdout launcher already owns this candidate.\n' >&2; exit 75; }
if ! flock -n "$PAIRED_RUN/paired-scheduler.lock" true; then
  printf 'The existing experiment scheduler is running; do not launch another.\n' >&2
  printf 'Follow its log: tail -n 80 -F "%s/controller.log"\n' "$PAIRED_RUN" >&2
  exit 75
fi

PAIRED_PHASE=preflight
finish() {
  PAIRED_EXIT_CODE=$?
  trap - EXIT
  "$PAIRED_PYTHON" -c '
import datetime,json,os,sys
from pathlib import Path
root=Path(sys.argv[1])
code=int(sys.argv[3])
data={"timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "phase":sys.argv[2],"exit_code":code,"runner_pid":int(sys.argv[4]),
      "phase_locks":{phase:(root/(phase+"-lock.json")).is_file()
                     for phase in ("smoke","dev","holdout")}}
with (root/"dev-holdout-exits.jsonl").open("a") as history:
    history.write(json.dumps(data)+"\n")
temporary=root/"dev-holdout-exit.json.tmp"
temporary.write_text(json.dumps(data,indent=2)+"\n")
os.replace(temporary,root/"dev-holdout-exit.json")
' "$PAIRED_RUN" "$PAIRED_PHASE" "$PAIRED_EXIT_CODE" "$$" || true
  printf '[%s] pipeline_exit phase=%s exit_code=%s\n' "$(date -u +%FT%TZ)" "$PAIRED_PHASE" "$PAIRED_EXIT_CODE"
  exit "$PAIRED_EXIT_CODE"
}
trap finish EXIT
printf '%s\n' "$$" > "$PAIRED_RUN/dev-holdout-runner.pid"
printf '[%s] pipeline_start candidate=%s pid=%s\n' "$(date -u +%FT%TZ)" "$PAIRED_RUN" "$$"

"$PAIRED_PYTHON" -u -m scripts.paired_experiment preflight --output "$PAIRED_RUN"
for PAIRED_PHASE in smoke dev holdout; do
  printf '[%s] phase_start phase=%s\n' "$(date -u +%FT%TZ)" "$PAIRED_PHASE"
  "$PAIRED_PYTHON" -u -m scripts.paired_experiment run --output "$PAIRED_RUN" --phase "$PAIRED_PHASE"
  printf '[%s] phase_end phase=%s exit_code=0\n' "$(date -u +%FT%TZ)" "$PAIRED_PHASE"
done
