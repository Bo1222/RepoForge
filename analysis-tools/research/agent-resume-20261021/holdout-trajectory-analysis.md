# Holdout 轨迹分析（A3 vs A0）

数据来源：`report-holdout.json`、`analysis/t7/holdout/{facts.jsonl,labels.jsonl,stats.json,paired.json,trajectory-summary.json}`。2026-10-21 15:44 完成；clean 81 × (A3+A0) = 162/162 attempted/graded；19 个非 clean 未执行，保留 100 分母并单列。官方 Harness `not_run`。

## 1. 结果与效率

| 指标 | A3 | A0 |
| --- | ---: | ---: |
| resolved / 81 | **46** | **43** |
| cost | **$4.6247** | **$6.7176** |
| cost / resolved | **$0.1005** | **$0.1562** |
| 平均每 job | **$0.0571** | **$0.0829** |
| prompt tokens | **26.9M** | **40.7M** |
| completion tokens | **408k** | **603k** |
| generation wall mean | **310s** | **459s** |
| 工具步骤 mean | **20.4** | **30.7** |
| 有效验证率 | **51.9%** | **43.2%** |
| no-progress rate mean | **18.6%** | **33.6%** |

配对 81 对：both 40、neither 32、A3-only 6、A0-only 3；difference +3（+3.7pp）；bootstrap 95% CI [-3.7pp, +11.1pp]；符号检验 p=0.5078。准确率差异不显著；成本、token、wall time 明显下降。

## 2. 动作策略轨迹

A3 共 1755 次策略判断：

| decision | 次数 |
| --- | ---: |
| allow | 1606 |
| replan | 52 |
| verify | 59 |
| stop | 38 |

主要 reason：`eligible` 1576、`no_progress` 66、`verification_due` 55、`verification_already_attempted` 30、`need_public_test_target` 11、`verification_unavailable_or_failed` 9、`budget_reserve` 8。A3 执行 149 次 intervention，跳过 161 个模型提案。

Policy stop 的 38 次分布：

| stop reason | 次数 | 已解决 | 未解决 | 观察 |
| --- | ---: | ---: | ---: | --- |
| `no_progress` | 25 | 7 | 18 | 多数未解决；存在过早停止风险 |
| `verification_unavailable_or_failed` | 9 | 7 | 2 | 多数仍由独立判题解决 |
| `budget_reserve` | 4 | 1 | 3 | 预算末期停止，收益有限 |

A0 没有 policy stop，但 81 个 job 中 33 个以 `max_iterations` 结束；A3 将 33 次预算耗尽转化为 38 次策略停止，并把实际工具步骤均值从 30.7 降到 20.4。

## 3. 上下文构建轨迹

对 `trajectory.json` 中每个 `node.context_build` 汇总：

| 指标（81 job 聚合） | A3 | A0 |
| --- | ---: | ---: |
| builds / job mean | **19.3** | **26.5** |
| estimated input tokens mean | **15,470** | **17,154** |
| omitted sources / job mean | **20.8** | **32.5** |
| shortened sources / job mean | **3.3** | **2.0** |
| stale sources / job mean | **103.4** | **143.6** |
| deduplicated sources / job mean | **2.7** | **46.4** |
| jobs with omitted | 16 | 29 |
| jobs with shortened | 13 | 10 |
| jobs with stale | 74 | 72 |
| jobs with dedup | 17 | 17 |

A3 的轨迹更短，构建次数、平均输入 token、omitted/stale/dedup 都更低；A0 的重复读去重量显著更高（mean 46.4，最大 870），说明长预算轨迹会积累更多重复读取。A3 的 `shortened` 略高，但构建次数更少；`included_sources` 中 A3 额外有 358 次 `action_policy_feedback`。

**ContextBudgetExceeded 的 5 个 job（全部未解决）**：

- A0：`django-14351`、`sympy-23413`、`sympy-24443`
- A3：`django-16938`、`sympy-23262`

这些 job 以 `generation_error` 结束，报告中为 `model_protocol` 失败；其中 `sympy-24443/a0` 出现 132 个 omitted、237 个 dedup，`sympy-23262/a3` 出现 172 个 omitted、73 个 dedup。说明硬预算会明确失败而不是静默丢上下文，但必选内容/执行事实持续增长时仍可能耗尽容量；这是上下文管理的真实边界，不是“无限续跑”。

## 4. 成功/失败案例

| instance | 方向 | A3 finish / stop | A3 steps | A0 finish | A0 steps | 说明 |
| --- | --- | --- | ---: | --- | ---: | --- |
| django-12754 | A3-only | `policy_stop` / `budget_reserve` | 40 | `max_iterations` | 75 | A3 在预算末停止并保留可判题补丁；A0 继续到 75 步仍未解决 |
| sympy-24443 | A3-only | `terminal` | 17 | `generation_error` (ContextBudgetExceeded) | 37 | A0 上下文预算耗尽；A3 更短轨迹完成 |
| sympy-24562 | A3-only | `terminal` | 11 | `max_iterations` | 40 | A0 候选补丁语法错误（人工裁定未解决） |
| sphinx-10673 | A3-only | `tool_error` | 22 | `max_iterations` | 42 | A3 补丁仍被独立判题解决；A0 预算耗尽 |
| django-14007 | A0-only | `policy_stop` / `no_progress` | 19 | `max_iterations` | 44 | 典型过早停止：A3 在 19 步判断无进展，A0 继续后解决 |
| sympy-18698 | A0-only | `tool_timeout` | 35 | `terminal` | 32 | A3 候选补丁在 `test_sqf` 挂起；不是策略停止 |
| matplotlib-20859 | A0-only | `terminal` | 32 | `terminal` | 28 | A3 出现 edit_error，A0 解决；属于候选编辑差异 |

**策略可改进点**：`no_progress` 的 25 次停止中 18 次未解决；`budget_reserve` 的 4 次中 3 次未解决。可在 stop 前加入一次低成本定向验证或允许 1–3 轮 replan，优先修复 `django-14007` 这类 A0-only 案例；`verification_unavailable_or_failed` 的 9 次中 7 次已解决，说明“无法验证时停止”多数没有损失正确补丁。

## 5. 恢复、失败与人工裁定

- `attempts.jsonl` 中没有多尝试或 `resume=true` 的 holdout job；本轮没有触发崩溃恢复路径，不能把设计中的恢复能力描述为“holdout 中实际恢复 N 次”。
- 4 个人工裁定：`sphinx-10673/a0` 循环导入、`sphinx-9461/a0` 收集失败、`sympy-18698/a3` 测试挂起、`sympy-24562/a0` `IndentationError`；原始 grading 与裁定记录均保留。
- A3 failure labels：`context_omission=9`、`model_protocol=7`、`invalid_repeat=6`、`edit_error=5`、`test_timing=5`、`budget=3`；A0：`budget=25`、`test_timing=5`、`model_protocol=4`、`unverified_finish=2`、`invalid_repeat=1`、`context_omission=1`。
- A3 将失败从“预算耗尽”迁移到“策略停止/模型协议/上下文边界”；`context_omission` 在 A3 更高，不能把策略收益归因于上下文压缩。

## 6. 结论与边界

1. **A3 的主要收益是成本/步骤，而不是显著准确率**：+3 resolved、p=0.5078、CI 跨 0；cost/resolved -35.6%，工具步骤 -33.5%，prompt tokens -33.8%。
2. **A3 把预算耗尽变成策略停止**：A0 33 次 `max_iterations`，A3 0 次；38 次 policy stop 中 15 次最终解决。
3. **策略停止质量不均**：验证不可用类停止多数正确（7/9 解决），`no_progress` 与 `budget_reserve` 类多数未解决，是下一版策略的优先改进点。
4. **上下文管理确实减少了重复/过期内容**：A3 的 builds、input、omitted、stale、dedup 均低于 A0；但有 5 个 ContextBudgetExceeded、9 个 context_omission 标签，容量治理和“被省略内容可发现性”仍是明确限制。
5. **Harness 层面**：162/162 干净执行、4 个人工裁定、0 个 holdout resume、官方导出 81 patch/臂；说明正常执行、判题隔离与失败审计可用，但不能据此声称崩溃恢复在 holdout 中被实际触发。

## 7. 复现

```text
report-holdout.json
analysis/t7/holdout/facts.jsonl
analysis/t7/holdout/labels.jsonl
analysis/t7/holdout/stats.json
analysis/t7/holdout/paired.json
analysis/t7/holdout/trajectory-summary.json
runs/<job_id>/trajectory.json
runs/<job_id>/policy-decisions.json
```

官方 Harness 仍未运行，所有 resolved 均为 `local_resolved`。
