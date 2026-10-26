# RepoForge

English | [中文](README.zh.md)

RepoForge is a repository-level coding agent that takes an issue through code search, context assembly, editing, testing and patch delivery. It combines recoverable LangGraph execution, version-aware Managed Context, local code retrieval and independent evaluation.

The interactive Textual/Rich CLI supports chat, isolated Git worktrees, follow-up tasks, cancellation, recovery and patch export. Batch evaluation supports two complete configurations: A0 with ActionPolicy disabled, and O2 fixed (`a5`) with deterministic action control enabled. Both share the same model, context, tools, retrieval and environment; the interactive CLI currently supports policy-off sessions.

## Install and run

Python 3.10–3.13 is supported. Install from this repository:

```bash
git clone https://github.com/Bo1222/IssueAgent.git RepoForge
cd RepoForge
uv sync --frozen
cp .env.example .env
# Set the provider credentials needed for your model in .env.
uv run repoforge chat
uv run repoforge chat --repo /path/to/clean-repo
uv run repoforge run --repo /path/to/clean-repo --task-file task.md --export /tmp/task-export
uv run repoforge resume SESSION_ID
# Equivalent module entry point:
uv run python -m repoforge.cli --help
```

The default CLI model is `openai/deepseek-flash`, using `DEEPSEEK_API_KEY` with native thinking disabled. Coding sessions require a clean source repository; changes are made in a separate worktree. Tools execute with the current user's permissions. Worktrees isolate code changes but do not provide a process security sandbox.

Project configuration uses `.repoforge/` and `REPOFORGE_*` environment variables. Interactive sessions default to `~/.repoforge/cli`. See the [CLI guide](docs/interactive_cli.md) and [rename notes](docs/renaming.md) for existing installations.

## Architecture

- **Agent execution:** LangGraph phases for preparation, action generation, policy decisions, execution, observation recording and iteration commits; SQLite checkpoints and an outcome journal support recovery with configuration and workspace checks.
- **Tools:** structured arguments route to reading, search, editing, commands and tests; source versions and test receipts keep observations tied to the code they describe.
- **Managed Context:** budgeted context reconstruction, stale evidence detection, duplicate removal, current code views and paired tool calls/results. The configured input cap is 32,768 estimated tokens.
- **Retrieval:** local BGE-M3/FAISS, symbol and lexical search, lexical fallback and RRF fusion.
- **Evaluation:** freeze generated product patches before independent native grading; retain reports, provenance and documented adjudications.

See the [architecture](docs/architecture.md), [Loop and context implementation](docs/agent_execution_context.md), and [project presentation](docs/project_resume.md).

## Recorded evaluation

The internal evaluation combines 30 development and 50 holdout tasks from SWE-bench Verified, with one run per configuration on each task:

| Configuration | Native local resolved | Graded runs |
| --- | ---: | ---: |
| A0, policy off | 44/80 (55%) | 80/80 |
| O2 fixed, `a5` | 46/80 (57.5%) | 80/80 |

The paired point estimate is +2.5 percentage points, with a 95% bootstrap interval of [-5, +10] percentage points and exact McNemar p=0.7539. This does not establish a stable policy improvement. The 80 tasks include development data and a documented grading-rule correction; the official SWE-bench Harness was **not_run**.

O2's recorded context builds all fit the configured estimated input budget. Across each task's cumulative, deduplicated search and ViewCode observations, the first 5/10 files cover at least one reference-modified file on 88.75%/96.25% of tasks. These are trajectory-level file coverage metrics, not single-query vector Recall@K or causal component gains.

The [publication bundle](reports/o2-a0/README.md) preserves generated patches, original reports, source fingerprints and adjudications. [Evaluation details](docs/evaluation.md) retain the original splits, sensitivity analysis and limitations. [Experiment commands](O2_A0_EXPERIMENT.md) create new frozen candidates; historical results retain their original source identity.

## Development

Run offline checks without invoking a model:

```bash
uv run --frozen python -m pytest tests/test_renaming_compat.py tests/policy -q
uv run --frozen python -m pytest tests/flow/test_langgraph_flow.py tests/flow/test_durable_recovery.py -q
```

Provider integrations and index-dependent tests require their respective external services or local assets. See [CHANGELOG.md](CHANGELOG.md) for the current delivery.

License and author attribution are preserved in [LICENSE](LICENSE), source notices and [CITATION.cff](CITATION.cff). Code sources include [Moatless Tools](https://github.com/aorwall/moatless-tools).
