# O2 5 题对照基线（运行前冻结）

日期：2026-10-22。历史数据来自 T7 dev30；O2 结果尚未产生。

## 历史结果（同 5 个实例）

| 实例 | A0 | T7-A3（旧 O） | 旧 O 错误 |
|---|---|---|---|
| pytest-dev__pytest-7490 | ✗ $0.109 / 40 步 | ✗ $0.028 / 16 步 | P2P 回归 `test_errors_in_xfail_skip_expressions` |
| django__django-10914 | ✓ $0.175 / 41 步 | ✓ $0.065 / 17 步 | 旧 O 判题环境错误（`eval.sh` rc=128）；生成侧无测试 |
| sympy__sympy-12481 | ✓ $0.026 / 15 步 | ✓ $0.114 / 25 步 | 旧 O 官方 F2P `test_args` 仍失败，且改了官方测试文件 |
| scikit-learn__sklearn-14053 | ✓ $0.008 / 10 步 | ✓ $0.013 / 13 步 | 旧 O resolved；2 次 FindFunction no-hit，后恢复 |
| astropy__astropy-8872 | ✗ $0.036 / 15 步 | ✗ $0.028 / 13 步 | 旧 O resolved，但无关 verify target + 5 次重复搜索，$0.180 |

合计：

- A0：**3/5，$0.354，121 步**
- T7-A3：**3/5，$0.249，84 步**
- 旧 O（Wave2 a3）：**2/5，$0.339，116 决策**（django 环境错误；可比 4 题 2/4）

## 评估口径

- Primary：同一 5 实例的 paired resolved 数（O2 vs A0）。
- 环境错误单列；django 若仍为 `eval.sh` 环境错误，不计入“模型失败”，但也不能算 resolved。
- Secondary：cost/resolved、prompt/completion tokens、tool steps、verify 次数、重复动作。
- N=5，只能做方向性判断，不做显著性声明。
- “大幅上升”的操作性定义：O2 至少比 A0 多解 2 题（例如 5/5 vs 3/5，或 4/5 vs 2/5 的可比子集），或 resolved 不降且 cost/resolved 明显下降；否则按“没有大幅上升”处理。

## 旧 O 已修复的确定性缺口（O2 已加）

1. repeat/no-progress stop 先于验证 → O2 先 verify/repair。
2. `verification_state`/`introduced_failures` 未消费 → O2 用它们决定 repair/verify。
3. `policy_stop` 绕过 PUBLIC_VERIFIED Finish guard → O2 在 stop 前强制验证/修复；不把未验证补丁当作已证实。
4. 提交 patch 含测试改动 → O2 a5 product-only + last-known-good。
5. `discover_test_targets` 返回无关测试 → O2 `require_relation=True`。
6. FailureInterpreter 只看最后一个 observation → O2 用 verification snapshot 的 `introduced_failures` 作为 repair 理由。

## O2 结果出来后要逐题回答

- 是否 resolved？与 A0/A3 的配对差异。
- 若不是：第一个确定性错误点是什么？
  - 环境错误（不可由代码修复）→ 记录，停止。
  - 模型语义错误（明确看到失败测试但改不对）→ 不是逻辑漏洞；按用户规则不做 prompt 修复，停止。
  - 验证/策略逻辑漏洞（未验证就停止、目标无关、重复动作、补丁回退失效）→ 修。
  - 检索 no-hit（FindFunction/FindCodeSnippet）→ 若新 fallback 未触发或无效，修 fallback。
- 若是：成本/步骤是否可接受；是否存在“解出来但贵很多”的实例。
