# O2/a5 5 题结果与 A0 对照（2026-10-22）

运行：`experiments/results/O2_dev30_sanity_20261021T161545Z`
同 reference、同 5 个实例、`variant=a5`、`official_harness=not_run`。
脚本最后 `aggregate` 因“子集不是 T7 冻结 job matrix”退出 1；5 个 job 本身均已完成，
已补写 `sanity-aggregate.json` 局部汇总，不影响逐题结果。

## 总表

| 指标 | O2 a5 | A0 | T7-A3（旧 O） |
|---|---:|---:|---:|
| resolved | **3/5** | **3/5** | **3/5** |
| cost | **$0.300** | $0.354 | $0.249 |
| prompt tokens | **1,564,184** | 2,293,372 | 1,386,955 |
| completion tokens | 19,364 | 16,044 | 18,179 |
| tool steps | **86** | 121 | 84 |

逐题：

| 实例 | O2 a5 | A0 | T7-A3 | 说明 |
|---|---|---|---|---|
| pytest-7490 | ✗ $0.033 / 13 | ✗ $0.109 / 40 | ✗ $0.028 / 16 | P2P 回归未修复 |
| django-10914 | ✗（env error）$0.033 / 16 | ✓ $0.175 / 41 | ✓ $0.065 / 17 | `eval.sh` rc=128，baseline 原失败未复现 |
| sympy-12481 | ✓ $0.032 / 16 | ✓ $0.026 / 15 | ✗ $0.114 / 25 | **相对旧 O 的真实修复** |
| sklearn-14053 | ✓ $0.040 / 12 | ✓ $0.008 / 10 | ✓ $0.013 / 13 | 解出但成本约 A0 的 5 倍 |
| astropy-8872 | ✓ $0.162 / 29 | ✗ $0.036 / 15 | ✗ $0.028 / 13 | 解出但成本高 |

可比 4 题（排除 django 环境错误）：

- O2：**3/4**，$0.267
- A0：**2/4**，$0.179
- T7-A3：**2/4**，$0.184

## 评估结论

**没有“大幅度上升”。**

- 总 resolved 与 A0 持平（3/5 vs 3/5）；django 的环境错误抵消了 astropy 的 +1。
- 可比 4 题 +1 resolved（astropy），但成本比 A0 高约 49%。
- 相对旧 O：resolved 持平，成本高约 21%，prompt 多约 13%，steps +2；sympy 修好，astropy 略便宜，但 sklearn 变贵。
- 方向性收益存在（sympy、astropy），但不是大幅、也不是免费；N=5 不能做显著性结论。

## 轨迹审计：真正的逻辑漏洞

### 已修复 1：`base_public` passed 被误判为 `TESTED_PASS`

- sklearn 的 base_public 在 node 9/12 均 passed（return_code=0），但工作区测试文件被模型修改过，
  `test_origins` 标为 `modified`；`verification.snapshot` 因此返回 `TESTED_PASS`，
  policy 认为公共验证未通过，最后以 `verification_unavailable_or_failed` 停止。
- 修复：`moatless/evaluation/verification.py` 对 `test_source=base_public` 的 passed 记录
  优先返回 `PUBLIC_VERIFIED`；base_public 的运行时 patch 本来就用 `ignore_tests=True`。
- 影响：sklearn 这类“公共测试已过、只是工作区测试被改”的实例不再被误判；预期减少多余 verify/成本。
- 测试：`tests/evaluation/test_verification.py::test_base_public_pass_survives_workspace_test_edits`。

### 已修复 2：target 相关性只过滤 discovered，未过滤 latest/context

- django 的 final target 是 `tests/file_storage/tests.py`，与修改的
  `django/conf/global_settings.py` 无关；`require_relation` 只作用于
  `discover_test_targets`，没有过滤 `latest_targets`/context 候选。
- 修复：新增 `is_related_test`；`PolicyState` 选 target 时，非模型显式提出的 target
  必须具备名称/import 关系，否则跳过。模型显式提出的 test 仍受信任。
- 影响：避免无关 verify target（django 的这次验证）；没有相关 target 时回到
  `need_public_test_target` replan。
- 测试：`tests/policy/test_target_validity.py::test_is_related_test_uses_name_or_import`。

### 已修复 3：sanity 子集 aggregate 误用冻结矩阵

- `aggregate(output, "dev")` 调用 `load_manifest`，要求 T7 冻结的 A0-A4 job matrix；
  O2 子集只有 a5，导致 exit 1。
- 修复：`analysis-tools/wave2_dev_sanity.py` 捕获该 ValueError，写局部 `sanity-aggregate.json`。
- 现有 O2 输出已补写局部汇总。

## 剩余失败：为什么停止修复

- **pytest-7490**：final `repeated_assertion_failure`，base_public 在 node 7/9 均失败；
  官方 P2P `test_errors_in_xfail_skip_expressions` 仍失败。模型看到了失败测试但没有改对，
  这是语义修复能力问题，不是验证/策略逻辑漏洞；按“禁止 prompt 优化”的规则停止。
- **django-10914**：`eval.sh` rc=128，baseline `original_failure_reproduced=false`；
  这是判题环境错误，不是 agent patch 的证据；需要重新 provision/修复环境，代码层停止。
- **sympy/astropy**：official resolved=true，但 base_public 记录 TESTED_FAIL（整文件测试
  失败/error）。这是公共测试整文件与官方 F2P/P2P 子集不一致导致的保守判定；没有 hidden
  test patch 时无法精确区分，且结果已经 resolved，不再为此改 policy。
- **sklearn**：resolved，但成本高；修复 1 预期减少一次误判验证，是否降本需要下一轮实测。
- 未做任何 system prompt / feedback 文案 / 契约话术修改。

## 下一步（等待批准）

新 split dev30/holdout50 的时间/成本估算见
`analysis-tools/O2_RUN_ESTIMATE.md`：

- 顺序纯计算约 **51 h**；
- 4 路并行约 **13 h**，加失败/重试/人工检查预计 **2–4 天**；
- 2 路并行预计 **4–7 天**；
- 单路/严格顺序含 holdout adjudication 预计 **7–10 天**；
- 模型成本预计 **$10–20**，主要瓶颈是 provision/index/判题时间。

未获批准前不启动新 split 实验。

---

## 错误两题的轨迹定位（修复后 a5 续跑）

运行：`experiments/results/O2fix_wrong2_20261021T164927Z`

### pytest-7490：语义 P2P 回归 + 运行期测试环境不可用

- 轨迹：
  - node 5 修改 `src/_pytest/skipping.py`；
  - node 6 修改官方测试文件；
  - node 7/8/11/12 的 RunTests 全部返回：
    `ModuleNotFoundError: No module named 'pytest'`，
    `test_run.status=error`；agent 从未拿到真实 pytest 结果；
  - node 18 仍然 `VerifiedFinish`，提交 product-only patch。
- 官方判题：
  - F2P 2/2 通过；
  - P2P `testing/test_skipping.py::test_errors_in_xfail_skip_expressions` 失败。
  - 具体 traceback：
    `test_errors_in_xfail_skip_expressions.py` 里有
    `@pytest.mark.xfail("syntax error")`；
    patch 在 `pytest_runtest_makereport` 增加：
    `item._store[xfailed_key] = xfailed = evaluate_xfail_marks(item)`；
    这次重新求值让 `evaluate_condition` 抛出
    `Failed: Error evaluating 'xfail' condition`，
    变成 `INTERNALERROR`，而不是测试期望的 `ERROR test_syntax`。
- 结论：
  - 直接原因：**模型 patch 对无效 xfail 条件重复求值，制造 P2P 回归**；
  - 间接原因：agent 的 RunTests 运行环境无法 import pytest，
    自测完全失明；官方 eval.sh 会 `pip install -e .`，agent 的测试命令没有。
  - 这属于模型语义错误 + 运行期测试命令/环境契约问题；按“禁止 prompt/契约类优化”的规则，
    不再继续改。

### django-10914：判题环境损坏，不是 patch 错误

- 轨迹：
  - node 3 修改 `django/conf/global_settings.py`（`FILE_UPLOAD_PERMISSIONS=0o644`）；
  - node 16 修改 docs；
  - 所有 RunTests（node 8/12/18/20/21）都返回：
    `ModuleNotFoundError: No module named 'django'`；
  - target 相关性修复生效：final target 为相关的
    `tests/settings_tests/tests.py`，不再是上一轮的 `tests/file_storage/tests.py`。
- 官方判题：
  - `eval.sh` 返回 **128**；baseline `original_failure_reproduced=false`；
  - `eval.sh` 在 `git status` / `git show` 阶段失败：
    - `git status` 显示仓库中几乎所有文件为 staged deleted、同时全部为 untracked；
    - `git show` 尝试访问 `https://github.com/django/django.git`，
      报 `Could not resolve host: github.com`；
    - `set -euxo pipefail` 使脚本在 `pip install -e .` 和测试之前退出。
  - 因此官方测试根本没有执行，patch 没有获得可判定的结果。
- 结论：**这是 provision 出来的 django 仓库/git 环境损坏 + 无外网，导致不可判题；不是模型 patch 失败。**
  常见修复 `FILE_UPLOAD_PERMISSIONS=0o644` 已在 patch 中，但因环境无法验证。
  按规则不在代码层继续修，需要重新 provision/修复该环境。

### 对两个逻辑修复的影响

- `base_public` 分类修复只影响“base_public passed 但工作区测试被改”的情况（sklearn 类）；
  pytest/django 的 base_public 都是 error，修复不改变这两题。
- target 相关性修复在 django 上确实生效（target 变为相关测试），但环境错误仍使判题失败。
- 所以这两题的结果不能用来评价两个逻辑修复；要评价应看剩余 3 题，但已按用户要求停止。
