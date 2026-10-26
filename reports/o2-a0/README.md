# A0 / O2 fixed 最终评估发布包

本目录保留最终结论、生成补丁、冻结配置和必要裁定，供项目展示与复核；完整执行轨迹和测试输出保留在原候选。总体为80个不同任务、每题两臂一次，160次正式运行，smoke不混入。

| 配置 | 解决数 | 解决率 | 完成判题 |
| --- | ---: | ---: | ---: |
| A0（policy off） | 44/80 | 55% | 80/80 |
| O2 fixed（a5） | 46/80 | 57.5% | 80/80 |

正向点估计+2.5pp；exact McNemar p=0.7539，配对bootstrap区间[-5,+10]pp。未建立稳定策略增益。结果为项目内部数据划分和native判题，包含已记录规则修正，官方Harness未运行。

## 文件用途

- `combined.json`：80题总体、组件性能与79对敏感性。定位指标为任务内累计文件覆盖，不是向量检索Recall@K。
- `report-dev.json`、`report-holdout.json`：原始冻结报告，原样保留，可逐题查看结果、费用和证据指纹。
- `predictions-a0.jsonl`、`predictions-a5.jsonl`：各80条Agent生成的最终产品补丁；两臂分开供独立判题。不是参考产品补丁，也不表示已经提交或合并到上游。
- `protocol.json`、`source-manifest.json`：冻结模型、策略、预算、题集、索引和原始源码身份；未包含API凭据。
- `adjudications.json`：影响最终判定的裁定记录及原记录hash，包含django-10097例外；具体口径见[评估说明](../../docs/evaluation.md)。
- `provenance.json`：原候选位置、复制来源与不可混淆的验证范围。
- `source-locks/`：原阶段锁的原样副本，其证据路径对应完整原候选，不能把本精简目录当作完整candidate直接运行`validate_phase`。
- `checksums.json`：本发布包文件的SHA256清单，供检查文件完整性。

## 只读复核

普通查看直接读取本目录。需要重算组件和总体指标时，用[只读工具](../../analysis-tools/paired/analyze.py)指向完整原候选，命令见[运行说明](../../O2_A0_EXPERIMENT.md)。本目录的源码与报告指纹属于实际历史运行，整理目录不重新签署历史实验。

生成阶段与独立判题分离；参考文件仅在运行结束后的离线定位分析中使用。详细统计、恢复说明和限制见[评估文档](../../docs/evaluation.md)。
