# CLI 结构与执行流程

这个目录提供普通聊天与本地 Coding Agent 的命令行入口和交互界面。界面与批处理命令共用 `SessionService`，实际模型调用、工具执行和状态恢复复用项目已有的 Agent、LangGraphFlow 和 Workspace。

## 文件作用

| 文件 | 作用 |
| --- | --- |
| [__main__.py](__main__.py) | `python -m moatless.cli` 的入口，调用 `main.main()`。 |
| [main.py](main.py) | 解析 `chat`、`run`、`resume` 及配置参数，加载 `.env`；启动交互界面或执行批处理任务。 |
| [app.py](app.py) | Textual 界面：任务输入、斜杠命令菜单、对话与工具记录、会话选择、Diff 和补丁确认。业务操作交给服务层。 |
| [service.py](service.py) | 会话管理与执行协调：创建、加载、续聊、保存、取消、恢复检查、测试与用量汇总、补丁应用和导出。 |
| [environment.py](environment.py) | 普通聊天使用禁用文件和命令的 ChatEnvironment；代码任务在 worktree 中读写文件、执行 Shell，取消时终止命令进程组。 |
| [worktree.py](worktree.py) | Git 操作：检查来源仓库、创建独立 worktree、生成 Diff、计算补丁 hash、检查并应用补丁。 |
| [__init__.py](__init__.py) | Python 包标识与简短说明，无启动逻辑。 |

## 整体流程

```text
python -m moatless.cli
    → __main__.py → main.py
    → chat / resume：app.py 接收任务
      run / resume --headless：main.py 批处理
    → SessionService 创建或加载普通聊天 / 代码会话
    → LangGraphFlow 驱动 SessionAgent
    → 请求 LLM → 普通文本转 Finish / 解析工具调用
    → 显示答案 / 执行工具并返回结果
                   ↑                            │
                   └──────── 下一轮 ────────────┘
    → 完成 / 达到限制 / 取消 / 出错：保存状态并显示结果
```

1. **读取配置**：`main.py` 加载 `.env`、JSON 配置和命令行参数。省略 `--repo` 进入普通聊天，不默认绑定当前目录；模型默认取 `MOATLESS_MODEL`，未设置时使用 `deepseek/deepseek-chat`。
2. **准备会话**：新任务通过 `SessionService.create()` 组装模型、Agent、Workspace 和 Flow。普通聊天使用空内存仓库，仅开放 Finish；指定 `--repo` 才检查 Git 仓库并创建独立 worktree。恢复任务通过 `load()` 加载已有会话。只打开空聊天界面不会调用模型。
3. **启动执行**：界面或批处理调用 `SessionService.start()`。Flow 使用已有历史生成模型请求，CLI 允许普通文本回复，内部转为 Finish，答案只显示一次；代码模式另可执行读取、搜索、修改、新建、Shell 和测试工具。库的默认严格工具调用行为不变。
4. **反馈结果**：代码工具在实际 worktree 中执行，结果进入 Node 轨迹；工具事件写入日志并通知界面或批处理输出。界面事件不替代模型历史。
5. **保存与续聊**：保存会话配置、轨迹和 checkpoint；后续消息继续使用同一会话；代码会话保留 worktree。取消或恢复时检查执行状态，对未确认的副作用不会自动重放。
6. **检查与交付**：通过 `/diff`、`/tests`、`/usage` 检查结果，`/export PATH` 导出记录。修改默认保存在 worktree；`/apply` 预览并确认后才应用到来源仓库，不自动提交。

## CLI 之外的关键模块

| 模块 | 与 CLI 的关系 |
| --- | --- |
| [LangGraphFlow](../flow/langgraph_flow.py) | 驱动执行阶段、迭代限制、checkpoint 与恢复。 |
| [SessionAgent](../agent/session_agent.py) | 继承 ActionAgent，在工具执行过程中发送生命周期事件。 |
| [本地工具](../actions/local_coding.py) | 实现 CLI 使用的读、搜、改、建、Shell 和测试操作。 |
| [ToolCallCompletionModel](../completion/tool_call.py) | 提供工具定义并解析模型返回的工具调用。 |
| [BaseCompletionModel](../completion/base.py) | 构造请求，最终通过 `litellm.acompletion()` 调用 LLM API。 |

`--model` 经配置传入 `ToolCallCompletionModel`。例如 `deepseek/deepseek-chat` 对应的密钥从 `DEEPSEEK_API_KEY` 读取；也可在配置中用 `api_key_env` 指定密钥环境变量名。

会话默认保存在 `~/.moatless/cli/<会话ID>/`，包含 `session.json`、`state.json`、`events.jsonl`、`checkpoint.sqlite`；仅代码会话含 `worktree/`。`session.json` 的 `mode` 为 `chat` 或 `coding`，普通聊天的来源、基准与 worktree 路径为空；旧会话缺失 mode 时按代码会话加载。可通过 `--session-dir` 或 `MOATLESS_CLI_DIR` 更改保存位置。

## 启动示例

在项目根目录激活虚拟环境，然后运行：

```bash
source .venv/bin/activate

# 普通聊天，无需 Git 仓库
python -m moatless.cli chat --model deepseek/deepseek-chat

# 代码任务；目标仓库需已提交且干净
python -m moatless.cli chat --repo /实际目标仓库路径 --model deepseek/deepseek-chat

# 一次性任务，不打开交互界面
python -m moatless.cli run --repo /实际目标仓库路径 '读取代码、修复问题并运行测试'

# 恢复已保存会话
python -m moatless.cli resume SESSION_ID
```

上述是终端启动命令。进入界面后直接输入“你是什么模型？”等自然语言；`/model` 在本地查看当前配置的 API 模型，不调用模型。`/repo /实际目标仓库路径` 切换到独立的新代码上下文，`/repo --clear` 返回新聊天上下文，原会话保留在 `/sessions`。

交互界面内输入 `/` 查看命令，`/architecture` 查看模块与当前会话的对应关系；Enter 发送，Shift+Enter 换行，Ctrl+C 取消任务，Ctrl+Q 退出。

完整配置、恢复机制与使用限制见 [CLI 使用文档](../../docs/interactive_cli.md)。

真实 SWE-bench issue 的运行方法和完成判定见 [演示指南](../../docs/swebench_demo.md)，实际结果见 [pytest-7490 报告](../../experiments/reports/swebench_demo.md)。
