# Moatless Tools

[English](README.md) | 中文

Moatless Tools 是一个个人兴趣项目，用于探索如何借助大语言模型（LLM）修改大型现有代码库。作者认为，除了依靠 Agent 推理出解决方案，构建好的工具同样关键：这些工具应将合适的上下文放入提示词，并妥善处理模型的响应。

_论文 [SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement](https://arxiv.org/abs/2410.20285) 所使用的实现，请参阅 [moatless-tree-search](https://github.com/aorwall/moatless-tree-search)。_

## SWE-bench

作者使用 [SWE-bench 基准测试](https://www.swebench.com/) 验证这些想法。

- Claude 4 Sonnet：解决率 70.8%，每个实例成本 0.63 美元。

# 快速体验

## 本地执行路线与计划状态

Moatless 官方网页、FastAPI 后端和自定义 Docker 执行路线已退出。当前源码保留 Python 库/demo 入口、本地 AsyncioRunner、KubernetesRunner 与已有存储后端。

本地交互式 CLI 已接通 Textual/Rich → SessionService → LangGraphFlow，支持无需 Git 的普通聊天，以及独立 Git worktree 中的代码任务，支持读取、搜索、编辑、命令、测试、续聊、取消、恢复与补丁导出。使用与配置见[CLI 文档](docs/interactive_cli.md)。

```bash
uv run moatless chat --model deepseek/deepseek-chat
# 代码任务：
uv run moatless chat --repo /path/to/clean-repo --model deepseek/deepseek-chat
uv run moatless run --repo /path/to/clean-repo --task-file task.md
uv run moatless resume SESSION_ID
```

Enter 发送，Shift+Enter 换行；`/help` 查看命令。省略 --repo 直接聊天，/model 本地查看模型，/repo PATH 新建代码上下文；代码任务的来源仓库必须干净；修改默认留在 worktree，`/apply` 预览并确认后才应用到来源仓库。

CLI 本身不依赖 Docker。worktree 隔离代码修改，工具仍以当前账号权限运行，不提供进程或文件权限层面的安全沙盒。首批验收使用 Python/pytest 仓库，要求来源仓库干净。

## 从源码安装

克隆仓库后，使用 uv 安装依赖：

```bash
# 克隆仓库
git clone https://github.com/aorwall/moatless-tools.git
cd moatless-tools

# 使用 uv 安装
uv sync
```


复制 `.env.example` 为 `.env`，按需填写模型密钥，并设置用于本地配置与轨迹的 `MOATLESS_DIR`。默认 runner 为 `asyncio`，配置已删除的 `docker` runner 会明确报错。

# 代码示例

## 基础 Agent 流程

```python
from moatless.actions import Respond
from moatless.agent import ActionAgent
from moatless.completion.tool_call import ToolCallCompletionModel

completion_model = ToolCallCompletionModel(
    model="gpt-4.1-mini",
    temperature=0.0,
    model_api_key=""
)

agent = ActionAgent(
    completion_model=completion_model,
    system_prompt="You are a helpful assistant that can answer questions.",
    actions=[
        Respond()
    ]
)

observation = await agent.run_simple("Hello")

print(observation.message)
```

## 代码检查 Agent

[查看 Notebook 示例](notebooks/code_inspector_agent.ipynb)

# SWE-bench：补丁生成与独立判题

使用保留的 Python Flow/Manager 入口生成 Agent 补丁与轨迹；CLI 可通过 `/export` 或 `--export` 导出补丁与轨迹，有合法 `instance_id` 时额外生成标准 predictions。本地 pytest 通过属于开发反馈，不能代替官方 `resolved`。合法预测记录包含 `instance_id`、`model_name_or_path`、`model_patch`。

已有 predictions 文件可独立调用官方 Harness：

```bash
uv run python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path predictions.jsonl \
  --max_workers 1 \
  --run_id local-grading
```

官方标准本地判题需要 Docker，与已退出的网页及 Moatless 自定义镜像无关，见[官方快速开始](https://www.swebench.com/SWE-bench/guides/quickstart/)。官方判题标签和隐藏测试不得进入 Agent 运行上下文；补丁生成与判题拆开记录，未执行时明确标为**未判题**。`swebench` 所需第三方 Docker 依赖继续保留在锁文件中。

下列配置表说明已有 Flow/Manager 配置，不是交互式 CLI 用法。历史模型成绩来自上游，不代表当前源码已通过验收。

## 流程配置

FlowManager/EvaluationManager 可使用以下已有流程 ID：

| Flow ID | 响应格式 | 适用模型 | 默认模型 |
| --- | --- | --- | --- |
| swebench_tools | 函数调用 | 原生支持函数调用的模型 | gpt-4o-mini-2024-07-18 |
| swebench_tools_and_reasoning | 函数调用 | 原生支持函数调用的推理模型 | claude-sonnet-4-20250514 |
| swebench_react | ReACT | 不原生支持函数调用的开源模型 | openrouter/mistralai/devstral-small |
| swebench_react_reasoning | ReACT | 不支持函数调用的推理模型 | openrouter/deepseek/deepseek-r1-0528 |

## 模型配置

EvaluationManager 支持通过以下参数配置模型。

### 模型选项

- `model_id`：指定完整的模型配置 ID。这会替换整个模型调用配置，包括 temperature、最大 token 数以及其他设置。
- `litellm_model_name`：仅覆盖 [LiteLLM 模型名称](https://docs.litellm.ai/docs/providers)，保留流程或模型配置中的其他设置，例如 temperature 和最大 token 数。

### 已验证模型

项目为已验证的模型提供默认配置。其他模型也可能可用，但尚未经过充分测试。

这里的“已验证”表示模型已在 SWE-bench 的 [Verified Mini 子集](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini) 上测试并确认可用。

仅指定 `model_id` 时，使用以下配置：

| 模型 | 响应格式 | 消息历史格式 | 动作中包含思考 | Verified Mini |
| --- | --- | --- | --- | --- |
| claude-3-5-sonnet-20241022 | tool_call | messages | 否 | [46%](https://experiments.moatless.ai/evaluations/20250119_claude_3_5_sonnet_20241022_0_0_n_20_fmt_tool_call_verified_mini) |
| claude-3-5-haiku-20241022 | tool_call | messages | 否 | [28%](https://experiments.moatless.ai/evaluations/20250118_claude_3_5_haiku_20241022_0_0_n_20_fmt_tool_call_verified_mini) |
| gpt-4o-2024-11-20 | tool_call | messages | 是 | [32%](https://experiments.moatless.ai/evaluations/20250119_azure_gpt_4o_0_0_n_20_fmt_tool_call_thoughts-in-action_1_verified_mini) |
| gpt-4o-mini-2024-07-18 | tool_call | messages | 是 | [16%](https://experiments.moatless.ai/evaluations/20250118_gpt_4o_mini_2024_07_18_0_0_n_20_fmt_tool_call_thoughts-in-action_6_verified_mini) |
| o1-mini-2024-09-12 | react | react | 否（已禁用思考） | [28%](https://experiments.moatless.ai/evaluations/20250114_o1_mini_2024_09_12_0_0_n_20_fmt_react_hist_react_verified_mini) |
| deepseek/deepseek-chat | react | react | 否 | [36%](https://experiments.moatless.ai/evaluations/20250118_deepseek_deepseek_chat_0_0_n_20_fmt_react_verified_mini) |
| deepseek/deepseek-reasoner | react | react | 否（已禁用思考） | [50%](https://experiments.moatless.ai/evaluations/20250120_deepseek_deepseek_reasoner_None_n_20_fmt_react_verified_mini) |
| gemini/gemini-2.0-flash-exp | react | react | 否 | [38%](https://experiments.moatless.ai/evaluations/20250119_gemini_gemini_2.0_flash_exp_0_0_n_20_fmt_react_verified_mini) |
| openrouter/meta-llama/llama-3.1-70b-instruct | react | react | 否 | - |
| openrouter/meta-llama/llama-3.1-405b-instruct | react | react | 否 | [28%](https://experiments.moatless.ai/evaluations/20250119_openai_meta_llama_Meta_Llama_3.1_405B_Instruct_FP8_0_0_n_20_fmt_react_verified_mini) |
| openrouter/qwen/qwen-2.5-coder-32b-instruct | react | react | 否 | [32%](https://experiments.moatless.ai/evaluations/20250119_openai_Qwen_Qwen2.5_Coder_32B_Instruct_0_0_n_20_fmt_react_verified_mini) |

## 数据集划分

EvaluationManager 可使用以下已有数据集名称：

| 划分名称 | 说明 | 实例数 |
| --- | --- | --- |
| lite | Lite 数据集的全部实例 | 300 |
| verified | Verified 数据集的全部实例 | 500 |
| verified_mini | [MariusHobbhahn/swe-bench-verified-mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini)，SWE-bench Verified 的子集 | 50 |
| lite_and_verified_solvable | 同时存在于 Lite 和 Verified 数据集中，且至少有一次成功 SWE-bench 提交的实例 | 84 |
