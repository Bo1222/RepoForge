# O2 新 split dev30/holdout50 时间估算（待批准）

日期：2026-10-22。数据来自 T7 候选与 Wave2 5 题 sanity，不是新 split 实测。

## 规模

- 新 split：dev30 + holdout50 = **80 个实例**
- 对照臂：A0（冻结 T7 源码，policy off）与 O2（a5）
- 生成 job：80 × 2 = **160 个**
- 判题：80 个 baseline + 160 个 candidate = **240 次 eval.sh**
- 模型：`openai/deepseek-flash`，max_iterations=40，max_cost=$1
- 环境：native provision；T7 holdout 100 个环境实测 91 ready / 9 failed（9%）

## 实测基准

| 项目 | 实测 |
|---|---|
| T7 dev A0 generation | 30 jobs，3.10 h，均值 6.21 min/job |
| T7 dev A3 generation | 30 jobs，1.66 h，均值 3.33 min/job |
| T7 holdout A0 generation | 81 jobs，10.33 h，均值 7.65 min/job |
| T7 holdout A3 generation | 81 jobs，6.98 h，均值 5.17 min/job |
| Wave2 5 题 O sanity | 105.7–498.4 s/job（1.76–8.31 min） |
| T7 holdout provision | 100 个环境，10.3 h wall，91 ready |
| Wave2 5 题 index references | 5 个索引约 23 min（约 4.6 min/个） |
| T7 dev cost | A0 $0.076/job；A3 $0.046/job |
| T7 holdout cost | A0 $0.067/job；A3 $0.046/job |

## 分项估算（顺序执行，含 20% 重试余量）

| 阶段 | 估算 |
|---|---|
| Provision 80 个环境 | 8.2 h × 1.2 ≈ **9.8 h** |
| 建 80 个索引 | 6.1 h × 1.2 ≈ **7.3 h** |
| 160 个 generation | A0 80 个 ≈ 9.3 h；O2 80 个 ≈ 7–9 h；合计 ≈ **20 h** |
| 240 次判题 | 2–4 min/次 ≈ 8–16 h；取 12 h × 1.2 ≈ **14.4 h** |
| **顺序总计** | **≈ 51 h 纯计算**（约 2.1 天） |

并行假设：

- 4 路并行：约 **13 h** 纯计算，加上环境失败/重试/人工检查 → **2–4 天**
- 2 路并行：约 **26 h** → **4–7 天**
- 单路/严格顺序、含 holdout adjudication 与失败重跑 → **7–10 天**（与文档 1–2 周一致）

## 成本估算

- A0：30 × $0.076 + 50 × $0.067 ≈ **$5.6**
- O2：80 × $0.046–0.068 ≈ **$3.7–5.4**
- 合计 **$9–11**；加重试/环境重跑，预计 **$12–20**
- 主要瓶颈是 provision/index/判题时间，不是模型费用。

## 风险

- T7 holdout 的 9% provision 失败率；新 split 含 xarray/requests/seaborn 等仓库，按 10–15% 预留重试。
- 80 个环境加索引的磁盘需求文档估算 58 GB；当前 `/` 可用 565 GB。
- dev30 先跑、holdout50 后跑；holdout 结果不回改策略。
- 需要先创建 W2/O2 的 candidate manifest/controller（当前 a5 profile 已在代码里，但新 split 的 manifest/controller 尚未建立）。
