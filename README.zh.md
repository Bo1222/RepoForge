# Moatless Tools

[English](README.md) | 中文

Moatless Tools 是一个个人兴趣项目，用于探索如何借助大语言模型（LLM）修改大型现有代码库。作者认为，除了依靠 Agent 推理出解决方案，构建好的工具同样关键：这些工具应将合适的上下文放入提示词，并妥善处理模型的响应。

_论文 [SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement](https://arxiv.org/abs/2410.20285) 所使用的实现，请参阅 [moatless-tree-search](https://github.com/aorwall/moatless-tree-search)。_

## SWE-bench

作者使用 [SWE-bench 基准测试](https://www.swebench.com/) 验证这些想法。

- Claude 4 Sonnet：解决率 70.8%，每个实例成本 0.63 美元。

# 快速体验

## 使用 Docker 运行

1. 克隆仓库：

   ```bash
   git clone https://github.com/aorwall/moatless-tools.git
   cd moatless-tools
   ```

2. 设置环境变量：

   ```bash
   cp .env.example .env
   ```

   编辑 `.env` 文件，填写 API 密钥和其他配置，其中必须设置 `MOATLESS_DIR`：

   ```text
   MOATLESS_DIR=/path/to/your/moatless/data
   ```

   **说明**：`MOATLESS_DIR` 指定 Moatless 保存配置文件和执行轨迹数据的目录。该目录会作为数据卷挂载到 Docker 容器中。

3. 启动服务：

   ```bash
   make run
   ```

4. 在浏览器中访问 http://localhost 打开界面。

## 从 PyPI 安装

```bash
# 仅安装基础包
pip install moatless

# 安装包含 Kubernetes Runner 支持的版本
pip install "moatless[kubernetes]"
```

## 从源码安装

克隆仓库后，使用 uv 安装依赖：

```bash
# 克隆仓库
git clone https://github.com/aorwall/moatless-tools.git
cd moatless-tools

# 使用 uv 安装
uv sync
```

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

# 运行 SWE-bench 评测

运行评测前，需要准备：

1. 至少一个 LLM 提供商的 API 密钥，例如 OpenAI 或 Anthropic。
2. [Voyage AI](https://voyageai.com) 的 API 密钥，用于使用 SWE-bench 实例预先生成的向量存储。

## 验证运行环境

运行完整评测前，可以先执行一个 SWE-bench 实例，检查环境是否配置正确：

```bash
uv run python scripts/docker_run.py  --flow swebench_tools --model-id gpt-4o-mini-2024-07-18 --instance-id django__django-11099 --evaluation-name testing_setup
```

该脚本会使用指定模型处理一个 SWE-bench 示例实例。

结果保存在 `.moatless/projects/testing_setup` 中。

## 执行评测

评测脚本在 Docker 容器中运行评测：

```bash
python3 scripts/run_evaluation.py  --model gpt-4o-mini-2024-07-18 --dataset-split [dataset_split] --evaluation-name [evaluation_name]
```

必需参数：

- `--dataset-split`：使用的数据集划分。
- `--evaluation-name`：评测名称。

可选参数：

- `--model`：评测使用的模型，默认值为 `gpt-4o-mini-2024-07-18`，等同于 `--model-id`。
- `--model-id`：使用的模型配置 ID，会替换整个模型配置。
- `--litellm-model-name`：覆盖 LiteLLM 模型名称，同时保留其他模型设置。
- `--flow`：评测使用的流程，默认值为 `simple_coding`。
- `--num-parallel-jobs`：并行任务数，默认值为 `1`。

## 流程配置

可以通过 `--flow` 指定以下流程：

| Flow ID | 响应格式 | 适用模型 | 默认模型 |
| --- | --- | --- | --- |
| swebench_tools | 函数调用 | 原生支持函数调用的模型 | gpt-4o-mini-2024-07-18 |
| swebench_tools_and_reasoning | 函数调用 | 原生支持函数调用的推理模型 | claude-sonnet-4-20250514 |
| swebench_react | ReACT | 不原生支持函数调用的开源模型 | openrouter/mistralai/devstral-small |
| swebench_react_reasoning | ReACT | 不支持函数调用的推理模型 | openrouter/deepseek/deepseek-r1-0528 |

## 模型配置

两个评测脚本均支持通过以下选项灵活配置模型。

### 模型选项

- `--model-id`：指定完整的模型配置 ID。这会替换整个模型调用配置，包括 temperature、最大 token 数以及其他设置。
- `--litellm-model-name`：仅覆盖 [LiteLLM 模型名称](https://docs.litellm.ai/docs/providers)，保留流程或模型配置中的其他设置，例如 temperature 和最大 token 数。

### 已验证模型

项目为已验证的模型提供默认配置。其他模型也可能可用，但尚未经过充分测试。

这里的“已验证”表示模型已在 SWE-bench 的 [Verified Mini 子集](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini) 上测试并确认可用。

仅指定 `--model-id` 时，使用以下配置：

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

可以通过 `--dataset-split` 指定以下数据集划分：

| 划分名称 | 说明 | 实例数 |
| --- | --- | --- |
| lite | Lite 数据集的全部实例 | 300 |
| verified | Verified 数据集的全部实例 | 500 |
| verified_mini | [MariusHobbhahn/swe-bench-verified-mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini)，SWE-bench Verified 的子集 | 50 |
| lite_and_verified_solvable | 同时存在于 Lite 和 Verified 数据集中，且至少有一次成功 SWE-bench 提交的实例 | 84 |

## 使用示例

```bash
# 使用 Claude 3.5 Sonnet 的完整模型配置运行评测
python3 scripts/run_evaluation.py \
  --model-id claude-3-5-sonnet-20241022 \
  --flow swebench_tools_and_reasoning \
  --dataset-split verified_mini \
  --num-parallel-jobs 5

# 仅覆盖模型名称，保留流程中的其他模型设置
python3 scripts/run_evaluation.py \
  --litellm-model-name openrouter/qwen/qwen-2.5-coder-32b-instruct \
  --flow swebench_react \
  --dataset-split verified_mini \
  --num-parallel-jobs 5
```
