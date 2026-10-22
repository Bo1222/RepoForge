# Moatless Coding Agent Harness：证据与数据附录

简历条目见 [project-summary.md](project-summary.md)。本文件只记录可核查的数据、口径与边界，供面试追问和后续更新使用。

## 1. 实验矩阵

- 数据集：`princeton-nlp/SWE-bench_Verified`，`test` split。
- 划分：dev 30 + holdout 100（seed42，holdout 预冻结）。
- 冻结 job 定义：15 smoke + 5 CLI + 150 dev + 500 holdout = 670。
- holdout 只执行锁定两臂 A3（winner）+ A0（control）。
- clean-only 规则：scan clean 81 个实例执行 A3+A0 = 162 job；19 个非 clean 不建索引、不执行，保留在 100 分母并单列。
- 评测：`local_resolved` 使用官方 test patch、FAIL_TO_PASS/PASS_TO_PASS、native grading 与 `get_eval_report`；官方 Harness 状态 `not_run`，`official_resolved=null`。

## 2. Dev 五臂结果（T7，150 job）

来源：[report-dev.json](../../../experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z/report-dev.json)；A0–A4 同一 manifest、模型、环境、索引与判题口径，共用 Managed Context，变化的是 ActionPolicy。

| 臂 | resolved | selected | 解决率 | 参考费用 | 每 resolved 成本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0（policy off） | 15 | 30 | 50.0% | $2.2802 | $0.1520 |
| A1 | 16 | 30 | 53.3% | $2.7374 | $0.1711 |
| A2 | 14 | 30 | 46.7% | $1.2097 | $0.0864 |
| **A3** | **17** | **30** | **56.7%** | **$1.3783** | **$0.0811** |
| A4 | 15 | 30 | 50.0% | $2.5511 | $0.1701 |

- A3 vs A0：+2 resolved；每 resolved 成本 `$0.1520 → $0.0811`，按未舍入值计算降低 46.7%。
- A3 总参考费用降低 39.6%；实际工具步骤 `902 → 492`，减少 45.5%。每 resolved 成本不是每 job 平均成本。
- 每臂 29 个 `local_grading=completed`、1 个 `environment_error`；`sphinx-doc__sphinx-7889` 的 `local_resolved=null`，仍计入 30 题分母。
- 费用是项目参考估算，`provider_bill=not_available`；`official_harness=not_run`。每臂每题仅运行一次，报告 `hypothesis_status=insufficient_evidence`，不能声称显著提升，也不能归因于上下文压缩。
- 历史 T4/T5 A0 候选 `T4_T5_bge_a0_40iter_1usd_20261016T074244Z` 的 dev30 报告为 15/30、$1.6196、$0.1080/resolved；与 T7 不是同一候选，只作为历史 A0 参考，不作受控比较。

## 3. Holdout 阶段结果（clean81；完整100题说明待续）

简历条目按“**Holdout100 完整评测待续**”保留更新位置。以下是 clean-only 协议已有报告，81 题子集已完成，其余 19 题未执行；不将子集完成等同于 100 题全部执行。

来源：[report-holdout.json](../../../experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z/report-holdout.json)、[stats.json](../../../experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z/analysis/t7/holdout/stats.json)、[paired.json](../../../experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z/analysis/t7/holdout/paired.json)。2026-10-21 15:44 完成；clean 81 × (A3+A0) = 162/162 attempted/graded；19 个非 clean 未执行，保留 100 分母并单列。

| 指标 | A3 | A0 |
| --- | ---: | ---: |
| clean 81 已执行 | 81 | 81 |
| resolved | **46** | **43** |
| resolved / clean 81 | 56.8% | 53.1% |
| resolved / 冻结 100 | 46.0% | 43.0% |
| 参考费用 | **$4.6247** | **$6.7176** |
| 平均每 job | **$0.0571** | **$0.0829** |
| cost / resolved | **$0.1005** | **$0.1562** |

- 配对（81 对）：both 40、neither 32、仅 A3 解决 **6**、仅 A0 解决 **3**；difference +3（+3.7pp）；10,000 次配对 bootstrap 95% CI **[-3.7pp, +11.1pp]**，仅为描述性区间；精确符号检验双侧 **p=0.5078**。
- 仅 A3 解决：`django-12754`、`django-16612`、`sphinx-10673`、`sympy-15976`、`sympy-24443`、`sympy-24562`。
- 仅 A0 解决：`django-14007`、`matplotlib-20859`、`sympy-18698`。
- A3 参考费用降低约 31.1%，cost/resolved 降低约 35.6%；prompt tokens 26.9M vs 40.7M（约 -33.8%）；wall time mean 310s vs 459s（约 -32.4%）。
- A3 policy stops 38、budget stops 0；A0 policy stops 0、budget stops 33。effective verification rate 51.9% vs 43.2%；no-progress action rate 18.6% vs 33.6%。
- 失败标签：A3 `context_omission=9`、`model_protocol=7`、`invalid_repeat=6`、`edit_error=5`、`test_timing=5`、`budget=3`；A0 `budget=25`、`test_timing=5`、`model_protocol=4`、`unverified_finish=2`、`invalid_repeat=1`、`context_omission=1`。
- `hypothesis_status=insufficient_evidence`；`failure_labels_complete=true`；官方 Harness `not_run`，`official_resolved=null`。官方导出 holdout-a3/a0 各 **81** 条 patch，`missing_patches=38`（19 个非 clean × 2 臂，预期）。
- 口径：81 是 clean 子集，100 是完整冻结分母；resolved/100 含 19 个未执行实例，不能把 46/100 或 43/100 解释为 100 题上的完整成功率。上下文压缩本身仍无独立消融，不能把策略收益归因于上下文管理。
- 轨迹与上下文构建聚合见 [holdout-trajectory-analysis.md](holdout-trajectory-analysis.md) 与 [trajectory-summary.json](../../../experiments/results/T7_policy_ablation_40iter_1usd_20261017T170912Z/analysis/t7/holdout/trajectory-summary.json)：A3 policy stops 38（no_progress 25、verification_unavailable 9、budget_reserve 4），A0 budget stops 33；5 个 ContextBudgetExceeded（3 A0、2 A3）。

## 4. 人工裁定与失败处理

- `sphinx-10673/a0`：候选补丁导致 `sphinx.environment` 循环导入，pytest 收集失败；冻结判题器标为 `environment_error`。保留原 `grading.json` 到 `grading-attempts/`，新增 `grading-adjudication.json`，按用户确认写为 `completed/local_resolved=false`。
- `sphinx-9461/a0`：候选补丁删除 `get_object_members`，pytest 收集 ERROR（退出码 2）；同样人工裁定为未解决。
- `sympy-18698/a3`：候选补丁导致 `test_sqf` 在 600s/1800s 两次超时，目标 `test_factor_terms` 未运行；基线 `test_sqf` 通过，判为候选挂起，人工裁定为未解决。
- `sympy-24562/a0`：候选补丁留下重复且无函数体的 `if q == 0:`，pytest 导入时报 `IndentationError`；补丁可应用但候选无效，人工裁定为未解决。
- 失败尝试、原始 case/grading、测试输出与证据哈希均保留；官方 Harness 未运行。

## 5. Harness 组件与代码位置

| 组件 | 作用 | 位置/证据 |
| --- | --- | --- |
| NativeProvisioner | conda 环境 receipt、repo/base_commit 身份、依赖与证据 hash | `moatless/evaluation/native_provision.py` |
| NativeExecutor + Bubblewrap | 配置 native 隔离时限制文件系统/网络，管理超时与进程组；普通 CLI worktree 仅隔离代码改动 | [native_execution.py](../../../moatless/evaluation/native_execution.py)、[native_tools.py](../../../moatless/runtime/native_tools.py) |
| RepositoryIndexStore | 按 repo、完整 base commit、嵌入与切分配置标识共享 BGE-M3 索引并校验 receipt | [repository_store.py](../../../moatless/index/repository_store.py) |
| Managed Context | 任务导向上下文投影、token 预算、stale/dedup、执行事实 | `moatless/message_history/managed.py`、`builder.py`、`policy.py` |
| Execution Memory | test records、effective_status/superseded、workspace/environment version、failure signature | `moatless/execution_memory.py` |
| Durable/Recovery | SQLite 操作日志与业务快照、LangGraph checkpoint、workspace/runtime 指纹；未确认副作用报告 RecoveryConflict | [durable.py](../../../moatless/flow/durable.py)、[graph_state.py](../../../moatless/flow/graph_state.py) |
| 事件观测 | 追加式 events.jsonl / sdk.jsonl 留存事件与调用；不是仅凭 JSONL 恢复全部业务状态 | [详细恢复边界](project-summary.md#32-日志快照与外部副作用) |
| Native Grading | 官方 test patch/F2P/P2P、独立判题、原始失败复现 | `moatless/evaluation/native_grading.py` |
| Budget/Usage | 逐 job token/费用账本、cost guard、未知费用不按 0 | `moatless/evaluation/experiment_usage.py`、`budget.json` |
| Harness export/report | 官方 Harness 导出/对照工具 | `analysis-tools/export_official.py`、`report_official.py` |
| 历史回归/恢复 | 历史记录为 266 个测试、8 个崩溃窗口；本次未复核原始日志，未重新运行 | `tests/`、T4.2 recovery evidence、[CHANGELOG](../../../CHANGELOG.md) |

## 6. 边界与不可声称

- 官方 Harness 未运行，不能写“官方通过”或 `official_resolved`。
- M0 是 `offline-scripted/migration-mock` 行为基线，不是性能基线；[E1 manifest](../../../context_memory_evidence/e1/manifest.json) 状态为 `prepared_only`，无独立上下文任务分数。
- 本项目没有实现多 Agent/Skill 调度、分层摘要 + 长期向量记忆、SFT/RL 训练；不要套用模板中的这些能力。
- 人工裁定 job 必须在报告中披露；不能把 `environment_error` 原始输出静默改写。
- 当前 clean-only 协议已完成（162/162 clean job），不等于 100 题全部执行。简历中的 Holdout100 完整结果标注待续；已有子集数据以 `report-holdout.json` 与 `analysis/t7/holdout/` 为准，官方 Harness 未运行。
- 上下文方案是预算控制、确定性投影、截短与回读；中间诊断丢失、旧来源发现、必选内容增长与 tokenizer 估算边界见 [设计评估](project-summary.md#24-已识别的边界与优先改进)。不能声称无损语义压缩、无限续跑或成熟长期记忆。
- 本次仅更新说明并重算原始报告数字，没有调用 LLM、重跑基准、修改锁或结果。来源哈希与校验结果见 [project-summary-data-check.json](project-summary-data-check.json)。
