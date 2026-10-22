# O2/a5 final version verification (2026-10-22)

## Final scope

- Final implementation is **O2 / a5**: the model-facing protocol stays `code_and_test` (no acceptance plan, no ReplanRepair, no CheckSyntax, no ExplainTestConflict).
- One small policy fix: `ActionPolicy.decide` treats both `Finish` and `VerifiedFinish` as terminal. The production agent uses `VerifiedFinish`; before this fix the terminal branches (`no_product_patch`, finish-time verification) never fired.
- The pytest minversion problem is a **testbed** defect, not part of the answer patch. Option A re-provisions pytest-7490 with the current provisioner (`SETUPTOOLS_SCM_PRETEND_VERSION=6.0`) into a new environment root; the agent patch remains product-only.
- A6/A7 code remains in the working tree for analysis but is not used by the final O2 runs.

## Environment

- Reference: `experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z`.
- Old reused pytest baseline: `0.1.dev1+g7f7a36478` in `src/_pytest/_version.py`, `pyproject.toml minversion=2.0`; local RunTests aborted rc4.
- New environment: `/tmp/A7_reprovision_20261022T123738Z/environment`; new baseline/prefix; `src/_pytest/_version.py = 6.0`.
- Merge receipt: `/tmp/A7_reprovision_20261022T123738Z/provision-override.json`.
- `analysis-tools/reprovision_pytest_7490.py` performs the re-provision; `wave2_dev_sanity.py --provision-override` merges it. Historical T7 files are untouched.
- `experiment_native.apply_testbed_setup()` remains as a recorded fallback for old reused pytest workspaces (`testbed-setup.json`, `in_answer_patch=false`); the new receipt does not need it.

## O2/a5 bad cases

| instance | run | resolved | cost | steps | grading |
|---|---|---:|---:|---:|---|
| pytest-7490 | `O2b_a5_pytest_clean_repeat_20261022T134442Z` | **true** | $0.100707 | 35 | F2P 2/2, P2P 78/78 |
| pytest-7490 | `O2b_a5_bad2_20261022T132557Z` | false | $0.094527 | 35 | F2P 2/2, P2P 77/78 (`test_errors_in_xfail_skip_expressions`) |
| astropy-8872 | `O2b_a5_bad2_20261022T132557Z` | **true** | $0.041658 | 25 | F2P 1/1, P2P 80/80 |

- The pytest false/true difference is model trajectory variance: the false patch re-evaluated xfail marks in `pytest_runtest_makereport` for every phase; the true patch added `call.when == "call"`; the O2fix2 patch used `try/except Failed`. All three are product patches; the environment is identical.
- The `VerifiedFinish` terminal fix is active: in the false run node 32 the policy replaced a premature `VerifiedFinish` with a replan; it did not misjudge the patch.
- Astropy cost drops from O2 old $0.162146 / 29 to $0.041658 / 25; P2P 80/80.

## O2 final 5-instance run

`O2b_a5_final5_20261022T135343Z` — single run, 5/5 resolved, 0 errors, official Harness `not_run` (native SWE-bench-compatible grading).

| instance | resolved | internal completion | steps | tests | cost |
|---|---:|---|---:|---:|---:|
| pytest-7490 | ✓ | unverified `repeated_collection_error` | 33 | 3 | $0.105291 |
| django-10914 | ✓ | unverified `no_progress` | 13 | 1 | $0.014396 |
| sympy-12481 | ✓ | unverified `no_progress` | 26 | 2 | $0.070782 |
| sklearn-14053 | ✓ | **verified** (terminal) | 13 | 2 | $0.027635 |
| astropy-8872 | ✓ | unverified `verification_unavailable_or_failed` | 40 | 5 | $0.137508 |
| **total** | **5/5** | 1 verified / 4 unverified | **125** | **13** | **$0.355612** |

Tokens: 2,168,474 prompt / 36,332 completion.

Comparison with the O2 fixed combination (five separate historical per-task results, not one run):

| metric | O2 fixed combination | O2 final single run |
|---|---:|---:|
| resolved | 5/5 | **5/5** |
| reference cost | $0.377373 | **$0.355612** (−5.8%) |
| tool steps | 105 | 125 (+19%) |
| completion | not measured uniformly | 1 verified / 4 unverified |

The pytest final patch is product-only and contains no `_version.py`:

```diff
+    if xfailed is None and call.when == "call":
+        item._store[xfailed_key] = xfailed = evaluate_xfail_marks(item)
```

Official/native grading: F2P 2/2, P2P 78/78. The earlier clean pytest false run (`O2b_a5_bad2_20261022T132557Z`, $0.094527 / 35) used a different model patch without the `call.when` guard and failed P2P `test_errors_in_xfail_skip_expressions`; the repeated clean run and the final 5 run both resolved true. This is model trajectory variance, not an environment or terminal-policy error.

The four unverified internal statuses are O2 policy stops (`no_progress`, `repeated_collection_error`, `verification_unavailable_or_failed`); they do not change the independent native `local_resolved` result. The final version deliberately does not add A6 acceptance/plan/ReplanRepair gates; O2's action policy is not further optimized at the algorithm level.

## Verification

- `tests/policy` after the `VerifiedFinish` terminal change: **126 passed**, exit 0.
- `tests/evaluation/test_native_experiment.py` (including `test_testbed_setup_repairs_generated_pytest_version_without_answer_edit`): **13 passed**.
- O2 final run: `O2b_a5_final5_20261022T135343Z`; aggregate `sanity-aggregate.json` 5/5, 0 errors.
- The `O2b_a5_bad2_20261022T132557Z` astropy result: true, $0.041658 / 25, P2P 80/80.

