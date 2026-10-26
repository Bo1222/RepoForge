# A0 / O2 fixed 配对实验与复核

本入口实现`o2-a0-paired-v1`：两臂使用相同Graph、Managed Context、工具、模型、检索、环境和product-only导出，仅切换`action_policy.variant=off/a5`。当前最终80题结果见[评估说明](docs/evaluation.md)和[结果发布包](reports/o2-a0/README.md)。

## 题集与参数

数据为`princeton-nlp/SWE-bench_Verified`的test，采用[冻结Wave2清单](repoforge/evaluation/profiles/wave2_split.json)：dev30、holdout50不重叠。先执行5题双臂smoke（10 jobs），随后dev60、holdout100 jobs；正文合计80个正式实例、160次运行，smoke单列。

固定non-thinking DeepSeek Flash、temperature=0、主输出4096、40轮/$1项目参考软预算、模型180秒/工具120秒；Managed输入32768、观察4000、近期交互2组。具体模型与索引身份见[protocol.json](reports/o2-a0/protocol.json)。费用不是供应商账单，官方Harness未运行。

## 创建并运行新候选

需要Python依赖、Conda、Bubblewrap、本地BGE-M3服务，以及进程环境或`.env`中的模型凭据。公开运行工具配置和离线构建wheelhouse通过路径显式传入；它们不包含任务答案。

```bash
PAIRED_RUN=/absolute/path/to/new-candidate
PAIRED_CONDA=/absolute/path/to/conda
PAIRED_PUBLIC_TOOLS=/absolute/path/to/public-tools/profile.json
PAIRED_BUILD_WHEELS=/absolute/path/to/build-wheels

.venv/bin/python -m scripts.paired_experiment prepare --output "$PAIRED_RUN" \
  --environment-root "$PAIRED_RUN/environments" --conda "$PAIRED_CONDA" \
  --public-tools "$PAIRED_PUBLIC_TOOLS" --build-wheels "$PAIRED_BUILD_WHEELS"
.venv/bin/python -m scripts.paired_experiment audit --output "$PAIRED_RUN"

nohup bash scripts/run_o2_a0_dev_holdout.sh "$PAIRED_RUN" \
  >> "$PAIRED_RUN/dev-holdout.log" 2>&1 < /dev/null &
tail -n 80 -F "$PAIRED_RUN/dev-holdout.log"
```

顺序脚本执行preflight、smoke、dev、holdout，以子进程退出推进阶段，不轮询结果、不自动重采样，失败立即停止；dev须等smoke锁，holdout须等dev锁。已有调度器时拒绝重复启动。退出码、阶段锁状态写入`dev-holdout-exit.json`并追加到`dev-holdout-exits.jsonl`。

阶段运行也可使用`python -m scripts.paired_experiment run --output PATH --phase smoke/dev/holdout`；参数实际传入一个phase值。代码/模型变更需新候选，历史结果保留原源码身份。

## 故障恢复与判题例外

环境失败两臂一致记为`env_blocked`并保留分母。生成/判题错误保留副作用和原始记录，不自动重采样；核对后才显式`--resume-job JOB_ID`恢复。

必要的手动复核工具集中在[analysis-tools/paired/recovery](analysis-tools/paired/recovery/)：

- `retry_network_blocked.py`：仅处理符合规则且尚无答案的网络准备失败，`--apply`前可只读审阅。
- `preprovision_mamba.py`：记录环境求解与公开依赖适配，保持题目、模型和答案身份不变；需要Conda前缀中可用的mamba。
- `adjudicate_grading_case.py`、`adjudicate_baseline_collection_error.py`：对照同环境基线和实际测试报告，记录裁定并保留原输出。
- `adjudicate_django_f2p_correction.py`：仅处理已说明的django-10097规则例外，不是通用自动裁定。

这些工具默认将新裁定写入候选的`recovery-records/`，不在源码目录产生结果；均提供`--help`。它们不会作为Agent可调用工具。历史裁定见[adjudications.json](reports/o2-a0/adjudications.json)。

## 不调用模型的结果复核

完整原候选包含冻结报告、轨迹和本地数据集cache时，可重新计算总体结果、代码定位覆盖及敏感性：

```bash
.venv/bin/python analysis-tools/paired/analyze.py \
  --candidate /absolute/path/to/completed-candidate \
  --output /absolute/path/to/new-analysis.json
```

也可用`--reference-dataset /path/to/swe-bench_verified-test.arrow`指定本地参考数据。参考文件名仅用于冻结后的离线分析，不提供给生成Agent。工具不安装环境、不调用模型或运行测试；原报告保持只读。

[结果发布包](reports/o2-a0/README.md)包含最终指标与生成补丁，不包含全部原始运行目录；其`source-locks/`是原运行锁的副本，完整`validate_phase`须对完整原候选执行。发布包自校验使用`checksums.json`。

## 离线回归检查

```bash
.venv/bin/python -m pytest tests/evaluation/test_native_provision.py \
  tests/evaluation/test_native_source_cache.py tests/evaluation/test_paired_experiment.py \
  tests/evaluation/test_paired_launcher.py tests/evaluation/test_o2_recovery_adaptations.py \
  tests/evaluation/test_native_experiment.py tests/evaluation/test_generation_worker.py \
  -q -o log_cli=false
```

这些检查使用固定响应或临时fixture，不启动真实模型实验。
