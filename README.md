# Moatless Tools
English | [中文](README.zh.md)

Moatless Tools is a hobby project where I experiment with some ideas I have about how LLMs can be used to edit code in large existing codebases. I believe that rather than relying on an agent to reason its way to a solution, it is crucial to build good tools to insert the right context into the prompt and handle the response.

_For the implementation used in the paper [SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement](https://arxiv.org/abs/2410.20285), please see [moatless-tree-search](https://github.com/aorwall/moatless-tree-search)._

## SWE-Bench
I use the [SWE-bench benchmark](https://www.swebench.com/) as a way to verify my ideas. 

* Claude 4 Sonnet - 70.8% solve rate, $0.63 per instance.

# Try it out

## Local execution roadmap

The official Moatless web UI, FastAPI backend and custom Docker execution route have been retired. The current checkout supports Python library/demo entrypoints, local AsyncioRunner, KubernetesRunner and existing storage backends.

The local Textual/Rich CLI uses a shared SessionService, LangGraphFlow, with ordinary chat or an isolated Git worktree for coding. It supports reading, search, editing, commands, tests, follow-up tasks, cancellation, recovery, and patch export. See the [CLI guide](docs/interactive_cli.md).

```bash
uv run moatless chat --model deepseek/deepseek-chat
# Coding tasks:
uv run moatless chat --repo /path/to/clean-repo --model deepseek/deepseek-chat
uv run moatless run --repo /path/to/clean-repo --task-file task.md
uv run moatless resume SESSION_ID
```

Enter sends; Shift+Enter inserts a newline; `/help` lists commands. Omit --repo for ordinary chat; /model displays the configured model locally; /repo PATH starts a new coding context. Coding requires a clean source repository. Changes stay in the worktree until `/apply` previews the patch and asks for confirmation.

The CLI does not depend on Docker. Worktrees isolate code changes; tools run with the current user's permissions and do not provide a process or filesystem security sandbox. Initial acceptance targets Python/pytest repositories and requires a clean source repository.

## Install from source

Clone the repository and install using uv

```bash
# Clone the repository
git clone https://github.com/aorwall/moatless-tools.git
cd moatless-tools

# Install using uv
uv sync
```


Copy `.env.example` to `.env`, set provider keys as needed, and set `MOATLESS_DIR` for local configuration and trajectories. The default runner is `asyncio`; selecting the retired `docker` runner raises an explicit error.

# Code Examples

## Basic agent flow

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

## Code inspector agent

[See notebook](notebooks/code_inspector_agent.ipynb)

# SWE-bench: patch generation and independent grading

Generate agent patches and trajectories through the retained Python Flow/Manager entrypoints; the CLI exports patches and trajectories with `/export` or `--export`, and standard predictions when a valid `instance_id` is supplied. A local pytest pass is development feedback, not official `resolved` evidence. Valid prediction records contain `instance_id`, `model_name_or_path` and `model_patch`.

Run the official Harness separately on an existing predictions file:

```bash
uv run python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path predictions.jsonl \
  --max_workers 1 \
  --run_id local-grading
```

The standard local official Harness requires Docker; it is independent of the retired web UI and custom Moatless images. See the [official quickstart](https://www.swebench.com/SWE-bench/guides/quickstart/). Keep official labels and hidden tests out of the agent's context. Record patch generation and grading separately; when grading has not run, mark it **not graded**. Third-party Docker dependencies required by `swebench` remain in the lock file.

The configuration tables below describe existing Flow/Manager configuration, not an interactive CLI. Historical model results are upstream results, not validation of this checkout.

## Flows 
Existing flow IDs available to FlowManager/EvaluationManager:

| Flow ID | Format | Best Suited For | Default Model |
|---------|--------|-----------------|---------------|
| swebench_tools | Function calling | Models with native function calling support | gpt-4o-mini-2024-07-18 |
| swebench_tools_and_reasoning | Function calling | Reasoning models with native function calling support | claude-sonnet-4-20250514 |
| swebench_react | ReACT | Open source models without native function calling support | openrouter/mistralai/devstral-small |
| swebench_react_reasoning | ReACT | Reasoning models without function calling support | openrouter/deepseek/deepseek-r1-0528 |

## Model Configuration

EvaluationManager supports model configuration through these parameters:

### Model Options

- `model_id`: Specify a complete model configuration ID. This replaces the entire completion model configuration including temperature, max tokens, and all other settings.
- `litellm_model_name`: Override only the [LiteLLM model name](https://docs.litellm.ai/docs/providers) while keeping all other completion model settings (temperature, max tokens, etc.) from the flow or model configuration.

### Verified Models

Default model configurations are provided for verified models. Note that other models may work but have not been extensively tested. 
Verified models are models that have been tested and found to work with the [Verified Mini subset](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini) of the SWE-Bench dataset.

When specifying just the `model_id` argument, the following configurations are used:

| Model | Response Format | Message History | Thoughts in Action | Verified Mini |
|-------|----------------|-----------------|-------------------|---------------|
| claude-3-5-sonnet-20241022 | tool_call | messages | no | [46%](https://experiments.moatless.ai/evaluations/20250119_claude_3_5_sonnet_20241022_0_0_n_20_fmt_tool_call_verified_mini) | 
| claude-3-5-haiku-20241022 | tool_call | messages | no | [28%](https://experiments.moatless.ai/evaluations/20250118_claude_3_5_haiku_20241022_0_0_n_20_fmt_tool_call_verified_mini) |
| gpt-4o-2024-11-20 | tool_call | messages | yes | [32%](https://experiments.moatless.ai/evaluations/20250119_azure_gpt_4o_0_0_n_20_fmt_tool_call_thoughts-in-action_1_verified_mini) |
| gpt-4o-mini-2024-07-18 | tool_call | messages | yes | [16%](https://experiments.moatless.ai/evaluations/20250118_gpt_4o_mini_2024_07_18_0_0_n_20_fmt_tool_call_thoughts-in-action_6_verified_mini) |
| o1-mini-2024-09-12 | react | react | no (disabled thoughts) | [28%](https://experiments.moatless.ai/evaluations/20250114_o1_mini_2024_09_12_0_0_n_20_fmt_react_hist_react_verified_mini) |
| deepseek/deepseek-chat | react | react | no | [36%](https://experiments.moatless.ai/evaluations/20250118_deepseek_deepseek_chat_0_0_n_20_fmt_react_verified_mini) |
| deepseek/deepseek-reasoner | react | react | no (disabled thoughts) | [50%](https://experiments.moatless.ai/evaluations/20250120_deepseek_deepseek_reasoner_None_n_20_fmt_react_verified_mini) |
| gemini/gemini-2.0-flash-exp | react | react | no | [38%](https://experiments.moatless.ai/evaluations/20250119_gemini_gemini_2.0_flash_exp_0_0_n_20_fmt_react_verified_mini) |
| openrouter/meta-llama/llama-3.1-70b-instruct | react | react | no | - |
| openrouter/meta-llama/llama-3.1-405b-instruct | react | react | no | [28%](https://experiments.moatless.ai/evaluations/20250119_openai_meta_llama_Meta_Llama_3.1_405B_Instruct_FP8_0_0_n_20_fmt_react_verified_mini) | - |
| openrouter/qwen/qwen-2.5-coder-32b-instruct | react | react | no | [32%](https://experiments.moatless.ai/evaluations/20250119_openai_Qwen_Qwen2.5_Coder_32B_Instruct_0_0_n_20_fmt_react_verified_mini) | - |

## Dataset splits

Existing dataset names for EvaluationManager:

| Split Name | Description | Instance Count |
|------------|-------------|----------------|
| lite | All instances from the lite dataset | 300 | 
| verified | All instances from the verified dataset | 500 | 
| verified_mini | [MariusHobbhahn/swe-bench-verified-mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini), a subset of SWE-Bench Verified  | 50 |
| lite_and_verified_solvable | Instances that exist in both lite and verified datasets and have at least one solved submission to SWE-Bench | 84 |
