# A7 implementation verification (2026-10-22)

## Scope

- Plan: `.claude/plans/plan-deep-dawn.md` (A7-simple: O2 single controller + A6 evidence/completion gate).
- Baseline commit: `2268485edb612a54e74f9c64cccdf72c84b4c4ea` (`2268485e`).
- Working tree already contained the O2/A6 changes before this task; this task edits on top of them and does not rewrite historical experiment results, locks or frozen answers.
- No model/API calls, no official Harness, and no 5-instance sanity run were executed by the agent.

## Implementation summary

- `moatless/policy/action.py`: `PolicyState`/`PolicyDecision` A7 fields; `extract_state(..., variant="a7")` consumes assessment, obligations, pending ReplanRepair gates and A6 progress facts; `ActionPolicy._decide_a7` keeps O2 thresholds/verification-first behavior and returns structured verify/replan requests.
- `moatless/policy/repair.py`: `assess(..., records=...)` reuses one execution snapshot; `CompletionAssessment.next_checks` is an ordered queue; receipt validity is the shared `verification_inputs.receipt_complete`.
- `moatless/policy/replanning.py`: `require_replan` creates O2-triggered evidence gates; `accept_replan` accepts any new evidence when a gate has no related files; `update_no_gain(..., open_gate=False)` persists facts only.
- `moatless/flow/langgraph_flow.py`: A7 pre-gates → O2 `ActionPolicy.decide` → execute → A6 `assess`; a7 initializes both `policy_state` and `repair_state`; stop goes through `PolicyDecision(kind="stop")`; finalize requires A6 ready/partial + O2 product/public verification + matching checkpoint; A6 checkpoint cannot resume as A7.  The production agent's `VerifiedFinish` is treated as terminal (a finish without a product patch stops; a finish with a patch is assessed), and `Reject` cannot bypass the completion gate.
- `moatless/evaluation/experiment.py`: `A7_PROFILES`, `prepare(repair_profile="a7")` writes `acceptance-repair-v3`; `make_jobs` selects the repair variant and rejects relabelled A6 profiles.
- `moatless/evaluation/generation.py`: A7 is product-only; checkpoint export gate covers a6/a7; `validate_completion_export` rejects a stale patch.
- `analysis-tools/a7_dev_sanity.py`: new native 5-instance sanity runner with the same T7 receipts and a `--dry-run` identity check.

## Commands and evidence

1. Target test set (plan Phase 5):
   - command: `.venv/bin/python -m pytest tests/policy/test_action_policy.py tests/policy/test_acceptance_repair.py tests/policy/test_six_group_contracts.py tests/policy/test_six_group_flow.py tests/policy/test_audit_boundaries.py tests/actions/test_t6_tool_contracts.py tests/actions/test_target_compile_commit.py tests/evaluation/test_generation_worker.py tests/evaluation/test_verification.py tests/evaluation/test_t4_t5_experiment.py -q`
   - result: **168 passed**, exit code `0` (243.38 s).
   - log: `target-tests.log`.
2. Policy suite (all A1-A7 policy/evidence tests):
   - command: `.venv/bin/python -m pytest tests/policy -q`
   - result: **123 passed**, exit code `0` (201.32 s).
3. Full offline suite:
   - command: `.venv/bin/python -m pytest -q`
   - result: **1114 passed, 5 failed**, exit code `1` (918.64 s); this full run started before the final a7-only progress-epoch tweak, which the final 123-test policy suite above covers.
   - log summary: `full-suite-summary.txt`.
   - the 5 failures are outside this change: `test_find_function`/`test_semantic_search` require unavailable DNS access to `stmoatless.blob.core.windows.net`; `test_execute_completion_rate_limit_retry` is in `moatless/completion/base.py`, which was already modified before this task and was not edited here.
4. A7 dry-run:
   - command: `.venv/bin/python analysis-tools/a7_dev_sanity.py --reference experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z --output <tmp> --only pytest-dev__pytest-7490,django__django-10914,sympy__sympy-12481,scikit-learn__scikit-learn-14053,astropy__astropy-8872 --dry-run`
   - result: variant `a7`, 5 jobs, A7 tool set, `acceptance-repair-v3`, manifest hash `8746ec1a0bdf4199b2b437172b2a852f8283eecda4d59f1186e6db0dc9cb5549`.
   - output: `dry-run.json`.

## A7 5-instance sanity runs

All runs issued after the user's explicit authorization; they use the historical five T7 smoke instances and reuse the T7 receipts/environments. Official Harness remains `not_run`.

| run | instances | resolved | cost | steps | completion |
|---|---|---|---|---|---|
| `A7_dev_sanity_20261022T085631Z` | 5 | 4/5 | $0.041667 | 60 | 5/5 unverified |
| `A7_dev_sanity_20261022T093336Z` | 5 | 4/5 | $0.100483 | 66 | 5/5 unverified |
| `A7_dev_sanity_20261022T101522Z` | 5 | 4/5 | $0.123481 | 85 | 5/5 unverified |
| `A7_bad2_20261022T110346Z` | pytest, astropy | **2/2** | $0.305400 | 82 | 2/2 unverified |
| `A7_final5_20261022T113754Z` | 5 | pending | pending | pending | pending |

Bad-case-only result (O2 fixed baseline in `o2-fixed-baseline.json`):

- pytest-7490: **resolved true**, $0.092278 / 37 steps, `replan_evidence_exhausted`; O2 $0.110810 / 32.
- astropy-8872: **resolved true**, $0.213121 / 45 steps, `verification_unavailable_or_failed`; O2 $0.162146 / 29. Cost is 31% above O2; it is explained by acceptance plan/check dispatch, base/current comparison and the repair cycle, while the old 24 ViewCode/8 search loop and `max_iterations` no longer occur.

Controller fixes discovered from these runs:

1. Pending ReplanRepair must allow bounded reads and must not let O2 re-replan the evidence read.
2. ReplanRepair may cite background rows plus at least one post-trigger new row; repository paths are resolved to the latest post-trigger read row.
3. A check whose target is not an executable test module is not an immutable obligation; the model may revise the plan for it.
4. Automatic check attempts are bounded per request, then return to the model for a plan/test revision before `verification_unavailable`.
5. An accepted ReplanRepair opens `repair_trial_open`, allowing the final repair edit/test instead of being re-blocked by O2 no-gain.
6. A repair gate is not created before an acceptance plan exists.
7. First test failure follows O2 semantics: one corrective edit is allowed, then `needs_check`; repeated failures create the A6 evidence gate. Creating the gate on the first failure blocked the correction and caused the astropy/pytest loops.

## Testbed environment defect and setup (pytest-7490)

- The reused native pilot environment for pytest-7490 was provisioned from an exact-base shallow fetch with no release tags. `setuptools-scm` generated `src/_pytest/_version.py = 0.1.dev1+g7f7a36478`, while `pyproject.toml` declares `minversion = "2.0"`. Every local `pytest` invocation aborts with `rc=4` before collecting tests.
- `native_provision.py` already declares the adaptation `SETUPTOOLS_SCM_PRETEND_VERSION=<instance.version>` for pytest/astropy/matplotlib/xarray, but the reused `T4_T5_native_environment_pilot_20261013` receipt predates it; its `install-repo.sh` has no such export and its baseline `_version.py` is still `0.1.dev1`.
- This is a **testbed** defect, not the issue's product behavior and not a change for the agent's answer patch. `experiment_native.apply_testbed_setup()` now repairs the generated version in the disposable job workspace before the agent starts, records `testbed-setup.json` with `in_answer_patch=false`, and leaves the product patch product-only.
- Evidence that the official/native grading is not blocked by this: final-run candidate `FAIL_TO_PASS` was 2/2 success; the single `PASS_TO_PASS` failure was `testing/test_skipping.py::test_errors_in_xfail_skip_expressions`, a real product regression from the wrong `skipping.py` patch. The bad-case `A7_bad2` patch included both `skipping.py` (correct hook point) and the `_version.py` workaround; it resolved, but the version edit must not be treated as the task fix.

## Option A: pytest-7490 re-provision

- New environment root: `/tmp/A7_reprovision_20261022T123738Z/environment` (historical T7 environment untouched).
- `analysis-tools/reprovision_pytest_7490.py` uses the current `NativeProvisioner`; the install command exports `SETUPTOOLS_SCM_PRETEND_VERSION=6.0`; cached conda/pip packages and the verified public source are reused.
- New baseline `src/_pytest/_version.py`: `__version__ = version = '6.0'` (before: `0.1.dev1+g7f7a36478`).
- Receipt: `/tmp/A7_reprovision_20261022T123738Z/pytest-reprovision.json`; mergeable receipt: `provision-override.json` (the first merge attempt exposed a receipt-shape key error; the corrected wrapper was used for the retry).
- Clean-environment pytest run: `experiments/results/A7_pytest_reprovision_20261022T123908Z`; resolved **false**, $0.078583 / 24 steps, `repair_exhausted`.
- Official/native grading of that run: F2P **2/2**, P2P **77 success / 1 failure** (`testing/test_skipping.py::test_errors_in_xfail_skip_expressions`). The local tests now execute (no minversion rc4) and the failure excerpt is the real P2P `INTERNALERROR`. The agent's patch still re-evaluates xfail marks in `pytest_runtest_makereport` instead of moving the re-evaluation to `pytest_runtest_call`; it exhausted its repair attempts. This is a model product-patch failure, not a controller false negative or an environment failure. The earlier bad2 result (`resolved=true`) used a different, correct patch *and* included the `_version.py` workaround; it is not the clean-environment result.

## Acceptance status

| # | Condition | Status |
|---|---|---|
| 1 | pytest no-gain/ReplanRepair; raw `minversion` feedback; resolved not below O2 | Code paths and offline Graph integration verified by `test_a7_no_gain_requires_replan_repair_before_writes`, `test_a7_raw_runner_output_enters_policy_feedback`, `test_a7_uses_the_same_verification_first_contract`; **actual 5-instance resolved/cost not_run** |
| 2 | astropy bounded ViewCode/search, cost not above O2 | no-gain/read-bound and ReplanRepair tests pass; **actual astropy trajectory/cost not_run** |
| 3 | django/sympy/sklearn mandatory checks carry `check_id` before unavailable | verified by `test_a7_mandatory_check_is_dispatched_with_check_id_and_finalizes` and `test_a7_policy_verification_appends_structured_check` |
| 4 | A7 5-instance resolved ≥ O2 fixed combination | **not_run** (requires user execution/approval) |
| 5 | target tests pass; no prompt text changes; official Harness `not_run`; A6b not mixed | 168 passed; prompt strings unchanged (A7 reuses the A6 branch); Harness `not_run`; A7 profile/protocol separate from a6 |
| 6 | new split only after A7 5-instance | **not started** (no provision/run) |

## Pending user-run command

```bash
.venv/bin/python analysis-tools/a7_dev_sanity.py \
  --reference experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z \
  --output experiments/results/A7_dev_sanity_$(date -u +%Y%m%dT%H%M%SZ) \
  --only pytest-dev__pytest-7490,django__django-10914,sympy__sympy-12481,scikit-learn__scikit-learn-14053,astropy__astropy-8872
```

Use a fresh output directory; the runner reuses T7 native receipts/environments and does not provision a new split. A6b results remain interim and are not combined with A7.

## Source hashes (sha256)

| file | sha256 |
|---|---|
| `moatless/policy/action.py` | `31b3f1d401125e0a48bd596a6577ffe786a1a773128de12bcb4b4f8a1e60d532` |
| `moatless/policy/repair.py` | `2e4e3e4a6bde9a9246fbd5dbf2152cd4b62f0b52cecd82f508db997029311f95` |
| `moatless/policy/replanning.py` | `77bdeca257173d1afcbedb8a301f304584ec905be17185c9bf16344d2a537379` |
| `moatless/flow/langgraph_flow.py` | `17a82e0a556b0bc95810e4c06e8c1e863a2349380ac9e22643c535d392eb42e5` |
| `moatless/evaluation/experiment.py` | `03e00bbee69e5a49d1a57b81c107ec986430a2ad8b4dd2a5a361e7ab6e4333c0` |
| `moatless/evaluation/generation.py` | `18d29f74f6cdfb5a74c636c7637b1e53c2b118e672cf45f662fd433d3a5e9c70` |
| `moatless/evaluation/verification.py` | `0219dfdd34064fffe33f35d57bd87e691d8d1faa821407808ca82a75d010af81` |
| `moatless/verification_inputs.py` | `6800af7acdf043453bb5a959d3b0473c5bd592a3604125fcbd67db77c6518c77` |
| `analysis-tools/a7_dev_sanity.py` | `f2e129ffb6f4874d38360864c3740f9531040cb71dbc081399fd175c49c98656` |

