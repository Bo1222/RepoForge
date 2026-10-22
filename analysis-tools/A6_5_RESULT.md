# A6 旧 5 题 sanity 结果（2026-10-22）

运行：`experiments/results/A6_dev30_sanity_20261022T044528Z`
variant=`a6`（`acceptance-repair-v2`），复用 T7 的 5 个 smoke 环境；
`official_harness=not_run`；耗时约 47 分钟。非正式 benchmark，只是逻辑/环境 sanity。

## 总表

| 指标 | A6 | O2 修复后组合* | O2 原始 5 题 | A0 | T7-A3 |
|---|---:|---:|---:|---:|---:|
| resolved | **4/5** | 5/5* | 3/5 | 3/5 | 3/5 |
| cost | **$0.438** | $0.377* | $0.300 | $0.354 | $0.249 |
| steps | 142 | 91* | 86 | 121 | 84 |
| prompt tokens | 2,384,648 | 1,683,514* | 1,564,184 | 2,293,372 | 1,386,955 |
| completion tokens | 32,290 | 19,977* | 19,364 | 16,044 | 18,179 |

\* O2 修复后组合不是一次 5 题运行：pytest/django 来自修复环境后的 2 题运行，
sympy/sklearn/astropy 来自修复前 O2 运行；用于上界对照，不能当作单次实验。
O2 原始 5 题是实际单次运行。

## 逐题

| 实例 | A6 | O2 | 结论 |
|---|---|---|---|
| pytest-7490 | ✗ `max_iterations` $0.168894 / 47 | ✓（修复后）$0.110810 / 32 | **A6 回归**；F2P 两项仍失败；34 次 FindCodeSnippet |
| django-10914 | ✓ `verification_unavailable` $0.010764 / 19 | ✓（修复后）$0.032341 / 16 | A6 更便宜但 completion 为 unverified |
| sympy-12481 | ✓ `verification_unavailable` $0.022886 / 18 | ✓（原始）$0.031908 / 16 | A6 更便宜但 unverified |
| sklearn-14053 | ✓ `verification_unavailable` $0.004611 / 12 | ✓（原始）$0.040167 / 12 | A6 更便宜但 unverified |
| astropy-8872 | ✓ `max_iterations` $0.230999 / 46 | ✓（原始）$0.162146 / 29 | A6 更贵、更慢且 unverified |

## A6 completion / acceptance 状态

- **5/5 completion 都是 `unverified`**，没有 `acceptance_satisfied`：
  - pytest：`max_iterations`，assessment `repair`；
  - django/sympy/sklearn：`verification_unavailable`，assessment `partial`，仍有 outstanding checks；
  - astropy：`max_iterations`，assessment `repair`，多条 unresolved failure obligations。
- 新工具使用情况：`RegisterAcceptancePlan` 有使用（pytest 2、django 1、sympy 2、sklearn 1、astropy 4），
  但 **CheckSyntax / ReplanRepair / ExplainTestConflict 均为 0 次**。
- A6 的验收计划/评估没有把测试执行推进到闭环：
  - pytest 的 `check_dynamic_xfail` RunTests rc4（`_version.py` 仍是 `0.1.dev1`，pyproject minversion 失败），
    模型继续 34 次 FindCodeSnippet，最后 40 轮耗尽；O2 修复环境后曾通过改 `_version.py` 让测试跑起来。
  - django/sympy/sklearn 只跑 workspace 测试（sklearn 有 base_public），
    assessment 仍缺 baseline comparison/public reproduction，最后 `verification_unavailable`。
  - astropy 3 次 RunTests 后进入 24 次 ViewCode/8 次 FindCodeSnippet，最后 46 轮耗尽；官方判题仍 resolved。

## 对 O2 的算法/逻辑增量（可写简历，但仅限设计层）

A6 相对 O2 的真实逻辑增量：

1. **验收义务账本**：`RegisterAcceptancePlan` 登记公开行为/必要检查/公开引文；控制器用绑定测试束、
   辅助输入、runner、环境和补丁版本的执行事实核销；模型不能自报 passed；跨版本失败义务保留；
   无关绿/过期绿不能核销。
2. **同输入反事实基线对照**：`verification_inputs.compare_runs` 在 base/current 同输入、同 runner、
   同环境下计算 introduced/preexisting/fixed；不可比保持 unknown，不制造空回归集。
3. **失败族修复预算与 ReplanRepair 门禁**：失败族身份与 target/hash 解耦；A→B→A 计两次；
   两次后要求新证据 replan，接收后仅最后一次修复；验证/修复预算分离。
4. **测试观察屏障 + 自动 finalize + VerifiedCheckpoint**：测试后隔离同一响应的待执行提议；
   全部验收满足自动完成并保存当前补丁 hash 的 checkpoint；导出当前产品补丁。
5. **目标解释器编译/提交一致性、deadline/partial/recovery、retry/上下文/overlay 支撑**：
   非执行 compile、源版本 token、写入前重查、原子替换；单 deadline/清理宽限/partial；
   合法 tool-call 配对与共享预算；当前 overlay 检索。

## 结论：这次 sanity 是否证明 A6 优于 O2

**没有。**

- **resolved**：A6 4/5；O2 修复后组合上界 5/5（非同一次运行），A6 少 1 题；
  pytest 是明确回归（O2 修复后 true，A6 `max_iterations`）。
- **成本/步骤**：A6 总成本 $0.438 > O2 修复后组合 $0.377 > O2 原始 $0.300；
  steps 142 > 91/86；prompt/completion 也更高。
  A6 在 django/sympy/sklearn 上更便宜，但这是 **`verification_unavailable` 提前停止**，
  不是完成验收后的成本节省；astropy 反而更贵。
- **证据质量**：A6 5/5 都是 `unverified`，没有 `acceptance_satisfied`；
  `CheckSyntax`/`ReplanRepair`/`ExplainTestConflict` 没被实际使用；
  验收义务没有闭环。A6 的设计更严格，但这次运行没有把它执行出来。
- **唯一可说的收益**：django/sympy/sklearn 的 `verification_unavailable` 早停让成本下降；
  这是“更早停止”，不是“更好验证/更高 resolved”。

## 简历口径建议

- **A6 算法设计**可以写在“算法/机制设计”或“离线合同验证”部分，
  但必须注明 5 题 sanity 为 4/5、5/5 unverified、pytest 回归、总成本 $0.438；
  不能写成已证明的性能提升。
- **性能数字**继续使用 O2 实测：
  - 旧 5 题单次 3/5 vs A0 3/5，可比 4 题 3/4 vs 2/4，$0.300 vs $0.354；
  - 修复后的错误两题 2/2 vs A0 1/2，$0.143 vs $0.284。
- A6 若要作为性能条目，需要先修 pytest 的 `_version.py`/minversion 反馈、让验收 plans 真正闭环，
  并在新 split 上做受控对照；当前不建议把 A6 写成“已提升 resolved”。
