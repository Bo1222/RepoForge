#!/usr/bin/env bash
# T4/T5 实验运行脚本。
#
# 默认动作(不带子命令 = run):**直接顺序执行完整实验**,脚本本身就是执行内容:
#   1  preflight(环境检查)                2  provision:5 题公共环境(smoke+CLI 共用)
#   3  公共工作区指纹预检(防目录/链接类启动错误)  4  组件验收(contracts/迁移/隔离/恢复/CLI 边界)
#   5  15 次 smoke(5 题 × loop/graph/a0)   6  5 次 CLI
#   7  native 门禁(dev 前置条件)            8  provision:dev 30 题公共环境
#   9  dev 30 题 BGE 索引(共享向量库,按实例建一次)
#  10  90 次 dev(30 题 × 3 次,a0)
# 每个 job 一条命令:run-phase = 生成 → 自有测试 → 独立判题 → 释放。
# 已完成(有 grading.json)的 job 自动跳过,可随时中断后重跑续传;
# 任一命令失败即停止(退出码非零),不自动重试。
#
# 用法:
#   scripts/run_t4_t5_experiment.sh [run] [候选目录]   # 直接执行完整实验(含 90 次 dev)
#   scripts/run_t4_t5_experiment.sh start [--check]    # 另一路径:后台启动带台账的调度器 controller.py
#   scripts/run_t4_t5_experiment.sh status             # 调度器模式的进度摘要
#   scripts/run_t4_t5_experiment.sh plan               # 流程与阶段完成情况
#   scripts/run_t4_t5_experiment.sh tail               # 跟随调度器日志
#   scripts/run_t4_t5_experiment.sh stop [--force]     # 停止调度器(--force 立即,在途 worker 跑完)
#
# 运行前要求:BGE 本地服务 127.0.0.1:8787 健康;仓库根 .env 配置 DEEPSEEK_API_KEY。
# 默认后台执行,无需 tmux:输出写入 <候选>/logs/run-<UTC>.log,用 tail 查看:
#   scripts/run_t4_t5_experiment.sh            # 后台启动,打印日志路径与 PID
#   tail -f <候选>/logs/run-*.log              # 查看进度(随时可关终端,不影响运行)
#   scripts/run_t4_t5_experiment.sh run --fg   # 前台执行(输出直接打在终端,Ctrl-C 可停)
# 候选目录默认读取 experiments/results/T4_T5_resume_20261015/candidate-path.txt。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
PYTHON="$ROOT/.venv/bin/python"
POINTER="$ROOT/experiments/results/T4_T5_resume_20261015/candidate-path.txt"

die() { echo "[错误] $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
用法: scripts/run_t4_t5_experiment.sh [子命令] [候选目录|选项]

  run(默认)        后台执行完整实验(含 90 次 dev);日志写入 <候选>/logs/run-<UTC>.log,
                   用 tail -f 查看;重复执行会跳过已完成 job 续传
  run --fg         前台执行(输出直接打在终端,不写日志文件)
  start [--check]   另一路径:后台启动带台账的调度器 controller.py(可关终端)
  status            调度器模式的进度摘要
  plan              完整流程与各阶段完成情况
  tail              跟随调度器控制台日志
  stop [--force]    停止调度器(--force 立即终止;在途 worker 跑完)

候选目录缺省取 experiments/results/T4_T5_resume_20261015/candidate-path.txt 的内容。
EOF
  exit 2
}

resolve_candidate() {
  local candidate="${1:-}"
  if [ -z "$candidate" ]; then
    [ -f "$POINTER" ] || die "找不到候选指针 $POINTER"
    candidate="$(cat "$POINTER")"
  fi
  candidate="$(readlink -f "$candidate")"
  [ -d "$candidate" ] || die "候选目录不存在: $candidate"
  [ -f "$candidate/controller.py" ] || die "候选目录缺少 controller.py: $candidate"
  printf '%s\n' "$candidate"
}

# 输出: "pid start_ticks alive(0/1) log"
controller_state() {
  "$PYTHON" - "$CANDIDATE" <<'PY'
import json, sys
from pathlib import Path
candidate = Path(sys.argv[1])
record = candidate / "launcher.json"
pid, ticks, alive, log = 0, "0", 0, ""
if record.exists():
    data = json.loads(record.read_text())
    pid, ticks, log = data.get("pid", 0), str(data.get("start_ticks", "0")), data.get("log", "")
    try:
        alive = 1 if Path(f"/proc/{pid}/stat").read_text().split()[21] == ticks else 0
    except OSError:
        alive = 0
print(pid, ticks, alive, log)
PY
}

# ---------- run 模式:直接顺序执行实验(默认动作) ----------
STOP=0

check_stop() {  # 收到停止请求后:不再启动任何新命令
  if [ "$STOP" = 1 ]; then
    echo; echo "[停止] 未启动新命令;重跑本脚本会自动跳过已完成 job。"
    exit 130
  fi
}

t4() {
  check_stop
  echo; echo "\$ python -m scripts.t4_t5_experiment $*"
  "$PYTHON" -m scripts.t4_t5_experiment "$@" --output "$CANDIDATE"
}

phase_jobs() {  # 按阶段列出 job_id
  "$PYTHON" - "$CANDIDATE" "$1" <<'PY'
import json, sys
from pathlib import Path
jobs = json.loads((Path(sys.argv[1]) / "jobs.json").read_text())
print(" ".join(job["job_id"] for job in jobs if job["phase"] == sys.argv[2]))
PY
}

dev_index_jobs() {  # 每个 dev 实例取一个代表 job(用于建索引)
  "$PYTHON" - "$CANDIDATE" <<'PY'
import json, sys
from pathlib import Path
jobs = [job for job in json.loads((Path(sys.argv[1]) / "jobs.json").read_text()) if job["phase"] == "dev"]
seen, first = set(), []
for job in jobs:
    if job["instance_id"] not in seen:
        seen.add(job["instance_id"]); first.append(job["job_id"])
print(" ".join(first))
PY
}

run_job() {  # 单个 job:已判题完成的跳过;独立会话运行(中断不会打断在途生成)
  local phase="$1" job="$2" rc=0
  check_stop
  if [ -f "$CANDIDATE/runs/$job/grading.json" ]; then
    echo "-- 跳过(已判题完成): $job"
    return 0
  fi
  echo; echo "\$ run-phase --phase $phase --job-id $job"
  setsid --wait "$PYTHON" -m scripts.t4_t5_experiment run-phase --phase "$phase" --job-id "$job" --output "$CANDIDATE" || rc=$?
  if [ "$STOP" = 1 ]; then
    echo "[停止] 已按中断请求退出;在途 job 会自行完成,重跑本脚本时自动跳过。"
    exit 130
  fi
  if [ "$rc" != 0 ]; then
    echo "[失败] job=$job(phase=$phase)退出码 $rc;已停止。"
    echo "        处理后重新运行本脚本:已完成 job 自动跳过,从失败处继续。"
    exit 1
  fi
}

cmd_run() {
  local mode="${1:-bg}"
  [ -x "$PYTHON" ] || die "缺少 $PYTHON(先创建 .venv 并安装依赖)"
  [ -f "$ROOT/.env" ] || echo "[警告] 未找到 $ROOT/.env(需要 DEEPSEEK_API_KEY)"
  set -a
  [ -f "$ROOT/.env" ] && . "$ROOT/.env" || true
  set +a
  [ -n "${DEEPSEEK_API_KEY:-}" ] || die "DEEPSEEK_API_KEY 未配置(写入 $ROOT/.env)"
  export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  local health
  health="$(curl -fsS --max-time 5 http://127.0.0.1:8787/health 2>/dev/null)" \
    || die "BGE 本地服务不可达(127.0.0.1:8787);先运行: .venv/bin/python -m scripts.local_bge.control start"
  case "$health" in
    *'"status":"ok"'*) ;;
    *) die "BGE health 异常: $health" ;;
  esac
  cd "$ROOT"
  mkdir -p "$CANDIDATE/logs"

  local pidfile="$CANDIDATE/logs/run.pid"
  if [ -s "$pidfile" ]; then
    local previous; previous="$(cat "$pidfile")"
    if kill -0 "$previous" 2>/dev/null && tr '\0' ' ' <"/proc/$previous/cmdline" 2>/dev/null | grep -qF "run_t4_t5_experiment.sh"; then
      die "已有运行中的 run 进程(PID=$previous);先 kill $previous,或等它结束"
    fi
  fi

  if [ "$mode" = "bg" ] && [ "${RUN_DETACHED:-0}" != 1 ]; then
    local log="$CANDIDATE/logs/run-$(date -u +%Y%m%dT%H%M%SZ).log"
    rm -f "$pidfile"
    RUN_DETACHED=1 nohup setsid bash "$SELF" run --bg "$CANDIDATE" >"$log" 2>&1 </dev/null &
    for _ in $(seq 1 30); do [ -s "$pidfile" ] && break; sleep 0.5; done
    echo "候选: $CANDIDATE"
    echo "已在后台启动,日志:"
    echo "  $log"
    echo "查看进度:  tail -f $log"
    if [ -s "$pidfile" ]; then
      echo "PID: $(cat "$pidfile")(停止: kill <PID>;当前 job 会跑完,重跑本命令自动跳过已完成 job)"
    else
      echo "[警告] 未捕获到 PID;如日志为空请查看该日志排查。"
    fi
    return 0
  fi

  if [ "${RUN_DETACHED:-0}" = 1 ]; then
    echo $$ >"$pidfile"
    echo "后台执行中(PID=$$),输出保存在重定向的日志文件里。"
  fi
  trap 'STOP=1; echo; echo "[中断] 不会再启动新 job;当前命令/在途 job 完成后退出。"' INT TERM
  echo "候选: $CANDIDATE"

  echo; echo "=== 1/10 preflight(环境检查)==="
  t4 preflight

  echo; echo "=== 2/10 provision:5 题公共环境(smoke+CLI 共用)==="
  t4 provision --phase smoke

  echo; echo "=== 3/10 公共工作区指纹预检(连续两次一致才继续)==="
  check_stop
  "$PYTHON" - "$CANDIDATE" <<'PY'
import json, sys
from pathlib import Path
from types import SimpleNamespace
from moatless.flow.durable import workspace_fingerprint
experiment = Path(sys.argv[1])
for identity, prepared in json.loads((experiment / "provision-native.json").read_text()).items():
    workspace = SimpleNamespace(repository=SimpleNamespace(repo_path=prepared["environment"]["repo"], shadow_mode=True),
                                runtime=None, _environment=None)
    first = workspace_fingerprint(workspace, str(experiment / "excluded-preflight.sqlite"))
    if first != workspace_fingerprint(workspace, str(experiment / "excluded-preflight.sqlite")):
        raise SystemExit(f"公共工作区指纹不一致: {identity}")
print("5 个公共工作区指纹一致")
PY

  echo; echo "=== 4/10 组件验收(contracts/迁移/隔离/恢复/CLI 边界)==="
  t4 contracts
  check_stop
  echo; echo "\$ python -m scripts.native_migration_audit"
  "$PYTHON" -m scripts.native_migration_audit --output "$CANDIDATE"
  t4 isolation
  t4 recovery
  t4 cli-boundaries

  echo; echo "=== 5/10 15 次 smoke(5 题 × loop_legacy/graph_legacy/a0)==="
  for JOB in $(phase_jobs smoke); do run_job smoke "$JOB"; done
  t4 report --phase smoke

  echo; echo "=== 6/10 5 次 CLI(pytest/django/sympy/sklearn/astropy)==="
  for JOB in $(phase_jobs cli_smoke); do run_job cli_smoke "$JOB"; done
  t4 report --phase cli_smoke

  echo; echo "=== 7/10 native 门禁(dev 前置条件)==="
  t4 gate

  echo; echo "=== 8/10 provision:dev 30 题公共环境 ==="
  t4 provision --phase dev

  echo; echo "=== 9/10 dev 30 题 BGE 索引(按实例构建一次,共享复用)==="
  for JOB in $(dev_index_jobs); do t4 prepare-index --phase dev --job-id "$JOB"; done

  echo; echo "=== 10/10 90 次 dev(30 题 × 3 次,a0)==="
  for JOB in $(phase_jobs dev); do run_job dev "$JOB"; done
  t4 report --phase dev

  echo; echo "全部完成。"
}


cmd_start() {
  local check_only="$1"
  [ -x "$PYTHON" ] || die "缺少 $PYTHON(先创建 .venv 并安装依赖)"
  [ -f "$CANDIDATE/manifest.json" ] || die "候选未冻结:缺少 $CANDIDATE/manifest.json"
  [ -f "$ROOT/.env" ] || echo "[警告] 未找到 $ROOT/.env(需要 DEEPSEEK_API_KEY)"
  set -a
  [ -f "$ROOT/.env" ] && . "$ROOT/.env" || true
  set +a
  [ -n "${DEEPSEEK_API_KEY:-}" ] || die "DEEPSEEK_API_KEY 未配置(写入 $ROOT/.env)"

  local health
  health="$(curl -fsS --max-time 5 http://127.0.0.1:8787/health 2>/dev/null)" \
    || die "BGE 本地服务不可达(127.0.0.1:8787);先运行: .venv/bin/python -m scripts.local_bge.control start"
  case "$health" in
    *'"status":"ok"'*) ;;
    *) die "BGE health 异常: $health" ;;
  esac

  if pgrep -f "$CANDIDATE/controller.py" >/dev/null 2>&1; then
    die "控制器已在运行(见 status/tail);如需重启先 stop --force"
  fi
  read -r PID TICKS ALIVE LOG <<<"$(controller_state)"
  if [ "$ALIVE" = 1 ]; then
    die "launcher 记录 PID=$PID 仍存活;用 status 查看"
  fi

  echo "候选:      $CANDIDATE"
  echo "manifest:  $("$PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1]))['manifest_hash'])" "$CANDIDATE/manifest.json")"
  echo "BGE:       ok"

  # 现场提醒:已开始生成但无冻结答案的 job 需要人工协调,控制器会在该 job 停止
  local pending
  pending="$("$PYTHON" - "$CANDIDATE" <<'PY'
import json, sys
from pathlib import Path
candidate = Path(sys.argv[1])
runs = candidate / "runs"
rows = []
if runs.is_dir():
    for run in sorted(runs.iterdir()):
        if (run / "generation-worker-started.json").exists() and not (run / "answer.json").exists():
            rows.append(run.name)
print(",".join(rows))
PY
)"
  [ -n "$pending" ] && echo "[警告] 需先协调的 job(已开始生成、无答案): $pending"

  if [ "$check_only" = 1 ]; then
    echo "[检查] 前置检查通过;移除 --check 即正式启动。"
    return 0
  fi

  mkdir -p "$CANDIDATE/logs"
  local log="$CANDIDATE/logs/controller-console-$(date -u +%Y%m%dT%H%M%SZ).log"
  rm -f "$CANDIDATE/pause-request.json"
  cd "$ROOT"
  nohup setsid "$PYTHON" "$CANDIDATE/controller.py" >>"$log" 2>&1 </dev/null &
  # setsid 可能因进程组状态重排 PID;按实际 cmdline 解析真实控制器 PID
  PID=""
  for _ in 1 2 3 4 5; do
    for candidate_pid in $(pgrep -f "$CANDIDATE/controller.py" || true); do
      if tr '\0' ' ' <"/proc/$candidate_pid/cmdline" 2>/dev/null | grep -qF "$CANDIDATE/controller.py"; then
        PID="$candidate_pid"; break
      fi
    done
    [ -n "$PID" ] && break
    sleep 1
  done
  if [ -z "$PID" ]; then
    echo "[错误] 未捕获到控制器进程,日志末尾:"
    tail -n 20 "$log" >&2
    exit 1
  fi
  TICKS="$(awk '{print $22}' "/proc/$PID/stat" 2>/dev/null || echo 0)"
  "$PYTHON" - "$CANDIDATE" "$PID" "$TICKS" "$log" <<'PY'
import datetime, json, sys
from pathlib import Path
candidate, pid, ticks, log = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
Path(candidate, "launcher.json").write_text(json.dumps({
    "pid": pid, "start_ticks": ticks, "log": log,
    "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
PY
  echo
  echo "已后台启动控制器 PID=$PID(关终端不影响)。"
  echo "日志:     $log"
  echo "看进度:   scripts/run_t4_t5_experiment.sh status"
  echo "跟日志:   scripts/run_t4_t5_experiment.sh tail"
  echo "停止:     scripts/run_t4_t5_experiment.sh stop"
}

cmd_status() {
  read -r PID TICKS ALIVE LOG <<<"$(controller_state)"
  "$PYTHON" - "$CANDIDATE" "$PID" "$ALIVE" "$LOG" <<'PY'
import json, sys, time
from pathlib import Path
candidate, pid, alive, log = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4]
PHASES = {"components": "组件验收", "smoke": "15 smoke", "cli_smoke": "5 CLI",
          "dev_preparation": "dev 环境与索引准备", "dev": "90 dev", "analysis": "分析待办"}
STATES = {"running": "运行中", "stopped_on_error": "已停止(错误,等待协调)",
          "analysis_pending": "运行完成,等待分析"}
total = {"smoke": 15, "cli_smoke": 5, "dev": 90}

print(f"候选:     {candidate}")
progress_path = candidate / "progress.json"
if not progress_path.exists():
    print("进度:     未启动(无 progress.json);先执行 start")
    raise SystemExit
p = json.loads(progress_path.read_text())
print(f"控制器:   {'运行中 PID=' + str(pid) if alive else '未运行'}")
print(f"状态:     {STATES.get(p.get('state'), p.get('state'))} | 阶段: {PHASES.get(p.get('phase'), p.get('phase'))}"
      f" | 更新: {p.get('timestamp', '')[:19]}")
done = p.get("completed", {})
print("完成:     " + " | ".join(f"{k} {done.get(k, 0)}/{total[k]}" for k in ("smoke", "cli_smoke", "dev")))

jobs = {}
jobs_path = candidate / "jobs.json"
if jobs_path.exists():
    jobs = {j["job_id"]: j for j in json.loads(jobs_path.read_text())}
if p.get("active_job"):
    job = jobs.get(p["active_job"], {})
    print(f"当前 job: {job.get('instance_id', '?')} {job.get('variant', '')} ({p['active_job']})")

ledger = p.get("live_ledger") or {}
if ledger:
    parts = [f"{k}={v}" for k, v in ledger.items() if not isinstance(v, dict)]
    if parts:
        print("现场:     " + ", ".join(parts))

rows = {}
vpath = candidate / "verification-jobs.jsonl"
if vpath.exists():
    for line in vpath.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[row["job_id"]] = row
if rows:
    solved = sum(1 for r in rows.values() if r.get("local_resolved") is True)
    cost = sum(r.get("reference_cost") or 0 for r in rows.values())
    print(f"已验收:   {len(rows)} 个 job(去重),本地解决 {solved} 个,生成估算 ${cost:.4f}")

coverage = candidate.parent.parent / "vector_store" / "development-coverage.json"
if coverage.exists():
    c = json.loads(coverage.read_text())
    print(f"dev 索引: {c.get('available_indexes', '?')}/30 就绪,{c.get('pending_indexes', '?')} 待建")

if p.get("latest_log"):
    print(f"最近日志: {candidate / p['latest_log']}")
if p.get("state") == "stopped_on_error":
    print(f"失败原因: {p.get('failure_reason')}")
    print(          "处理提示: 核对 stage-failures-*.jsonl 与 runs/<job>/ 现场;协调完成后再 start 续跑(不会自动重试)")

pending = []
runs = candidate / "runs"
if runs.is_dir():
    for run in sorted(runs.iterdir()):
        if (run / "generation-worker-started.json").exists() and not (run / "answer.json").exists():
            pending.append(run.name)
if pending:
    print(f"现场提醒: {len(pending)} 个 job 已开始生成但无冻结答案,需人工协调: {', '.join(pending)}")
PY
}

cmd_plan() {
  "$PYTHON" - "$CANDIDATE" <<'PY'
import json, sys
from pathlib import Path
candidate = Path(sys.argv[1])
def mark(ok): return "✅" if ok else "○ "
runs_n = 0
verified = {}
vpath = candidate / "verification-jobs.jsonl"
if vpath.exists():
    for line in vpath.read_text().splitlines():
        try: row = json.loads(line)
        except json.JSONDecodeError: continue
        verified[row["job_id"]] = row
def count(phase): return sum(1 for r in verified.values() if r.get("phase") == phase)
provision = candidate / "provision-native.json"
stages = [
    ("候选冻结(manifest/110 job 核定)", (candidate / "jobs-audit.json").exists()),
    ("preflight(conda/bwrap/git 环境检查)", (candidate / "preflight.json").exists()),
    ("provision:5 题公共环境", provision.exists()),
    ("公共工作区指纹预检(防 gitlink 目录错误)", (candidate / "fingerprint-preflight.json").exists()),
    ("首题索引复用确认(共享向量库)", (candidate / "index-references" / "pytest-dev__pytest-7490.json").exists()),
    ("组件验收(contracts/迁移/隔离/恢复/CLI 边界)",
     all((candidate / f"{name}-native").exists() for name in
         ("contracts", "migration", "isolation", "recovery", "cli-boundaries"))),
    (f"15 smoke(三种对照×五题) 已完成 {count('smoke')}/15", count("smoke") >= 15),
    (f"5 CLI 已完成 {count('cli_smoke')}/5", count("cli_smoke") >= 5),
    ("native 门禁(15+5 全验收后冻结)", (candidate / "native-gate.json").exists()),
    ("dev:30 题环境+索引齐备(30/30 后放行)", (candidate / "index-coverage-at-dev-start.json").exists()),
    (f"90 dev(30 题×3 次,并发 1) 已完成 {count('dev')}/90", count("dev") >= 90),
    ("报告 report-dev.json(失败/费用/判题汇总)", (candidate / "report-dev.json").exists()),
]
print(f"实验流程(候选 {candidate.name}):\n")
for i, (name, ok) in enumerate(stages, 1):
    print(f"  {mark(ok)}{i:>2}. {name}")
print("\n备注: ① 首题 pytest 属于 15 smoke,不重复计费;② smoke/CLI 与 90 dev 分开记录;")
print("      ③ 门禁未通过前不会 provision dev;④ 失败停止后由人工协调,再 start 续跑。")
PY
}

cmd_tail() {
  read -r PID TICKS ALIVE LOG <<<"$(controller_state)"
  if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
    LOG="$(ls -t "$CANDIDATE"/logs/controller-console-*.log 2>/dev/null | head -1 || true)"
  fi
  [ -n "$LOG" ] && [ -f "$LOG" ] || die "没有找到控制器控制台日志;先 start 或检查 $CANDIDATE/logs/"
  echo "跟随日志: $LOG(Ctrl-C 退出,不影响后台运行)"
  tail -n 50 -f "$LOG"
}

cmd_stop() {
  local force="$1"
  read -r PID TICKS ALIVE LOG <<<"$(controller_state)"
  touch "$CANDIDATE/pause-request.json"
  echo "已写入暂停标记 $CANDIDATE/pause-request.json"
  if [ "$ALIVE" = 1 ]; then
    echo "控制器 PID=$PID 将在当前 job 完成后停止调度(不会启动下一个 job)。"
    echo "恢复: scripts/run_t4_t5_experiment.sh start(会自动核验并跳过已完成 job)"
  else
    echo "控制器当前未运行;暂停标记会在下次 start 时自动清除。"
  fi
  if [ "$force" = 1 ]; then
    local target=""
    target="$(pgrep -f "$CANDIDATE/controller.py" | head -1 || true)"
    [ -n "$target" ] || die "未找到运行中的控制器进程"
    kill "$target"
    echo "已对控制器 PID=$target 发送 SIGTERM;在途 worker 会自行跑完并落盘(不会重复计费)。"
  fi
}

SUBCMD="${1:-run}"
if [ $# -gt 0 ]; then shift; fi
CANDIDATE_ARG=""; OPTION=""
for arg in "$@"; do
  case "$arg" in
    --check|--force|--fg|--bg) OPTION="${arg#--}" ;;
    *) CANDIDATE_ARG="$arg" ;;
  esac
done
case "$SUBCMD" in
  run)    CANDIDATE="$(resolve_candidate "$CANDIDATE_ARG")"; cmd_run "${OPTION:-bg}" ;;
  start)
    CANDIDATE="$(resolve_candidate "$CANDIDATE_ARG")"
    if [ "$OPTION" = "check" ]; then cmd_start 1; else cmd_start 0; fi
    ;;
  status) CANDIDATE="$(resolve_candidate "$CANDIDATE_ARG")"; cmd_status ;;
  plan)   CANDIDATE="$(resolve_candidate "$CANDIDATE_ARG")"; cmd_plan ;;
  tail)   CANDIDATE="$(resolve_candidate "$CANDIDATE_ARG")"; cmd_tail ;;
  stop)
    CANDIDATE="$(resolve_candidate "$CANDIDATE_ARG")"
    if [ "$OPTION" = "force" ]; then cmd_stop 1; else cmd_stop 0; fi
    ;;
  -h|--help|help) usage ;;
  *) usage ;;
esac
