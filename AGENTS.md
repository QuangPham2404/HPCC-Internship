# Repository Guidelines

## Project Structure & Module Organization

This repository documents GPU/HPC benchmark workflows for a PBS-managed cluster.

- `README.md` gives the project overview and confidentiality expectations.
- `PyTorch/` contains ResNet-50 synthetic Distributed Data Parallel benchmarks.
- `PyTorch/Containerized/` contains Apptainer-based PBS launch scripts and Python DDP scripts.
- `PyTorch/Baremetal/` contains module-based PBS launch scripts and matching Python DDP scripts.
- `user-scripts/vllm/single-node/` contains single-node vLLM smoke-test scripts and usage notes.
- `user-scripts/vllm/multi-node/` contains multi-node vLLM/Ray smoke-test scripts and usage notes.
- `HPL-HPCG/` and `vLLM/` currently contain placeholder README files.

There is no dedicated test directory or packaged application source tree.

## Build, Test, and Development Commands

No local build step is required. Validate edits with lightweight checks before submitting:

```bash
python -m py_compile PyTorch/Containerized/*.py PyTorch/Baremetal/*.py
bash -n PyTorch/Containerized/*.pbs PyTorch/Baremetal/*.pbs
bash -n user-scripts/vllm/single-node/*.pbs user-scripts/vllm/multi-node/*.pbs
```

Cluster execution is done through PBS:

```bash
qsub PyTorch/Containerized/singlenode-launch.pbs
qsub user-scripts/vllm/single-node/container_single_node_vllm_smoke_refined.pbs
```

Run PBS jobs only after replacing placeholder project, queue, container, model, and storage paths.

## Coding Style & Naming Conventions

Use clear, shell-compatible Bash in `.pbs` files with descriptive uppercase environment variables such as `BENCH_ROOT`, `TP_SIZE`, and `MASTER_PORT`. Prefer explicit comments for cluster-specific settings and keep placeholders generic, for example `<project>`, `<queue>`, and `<path>`.

Python scripts should use 4-space indentation, descriptive variable names, and standard PyTorch/DDP conventions. Keep benchmark parameters grouped near the top of scripts when possible.

## Testing Guidelines

There is no formal test framework or coverage requirement. For Python changes, run `py_compile`. For PBS script changes, run `bash -n` and review variable quoting, module names, resource requests, and cleanup handlers. For cluster behavior, use a small smoke test first, such as one node and one GPU.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, for example `Add baremetal single node launch script for vllm` and `Clean up misc`. Follow that style: describe the change directly in one concise line.

Pull requests should explain the benchmark workflow affected, list any required user edits, and note whether scripts were syntax-checked or run on the cluster. Do not include internal hostnames, IP addresses, credentials, scheduler account names, or confidential benchmark results.

## Security & Configuration Tips

Keep site-specific values out of committed files. Use placeholders for paths, PBS projects, queues, Hugging Face tokens, and internal networking details. When adding logs or outputs, ensure they do not expose node names, private addresses, or performance data that should remain internal.

## Permission instruction

For this directory, you are to READ ONLY by default. You only write and make change if my prompts indicate that you do so using command words such as "write", "edit", "make changes", "update", etc. Else, ask for permission to edit ANYTIME you want to write and you don't see me mention it

## Communication style

When working on a task, provide concise progress updates at important milestones:
- State what you are about to inspect or change.
- Report important findings, especially blockers, assumptions, or risky areas.
- After making changes, summarize what changed and what was not changed.
- After running tests or commands, summarize the result.

Do not narrate every command. Only do this for key milestones. Do not provide long internal reasoning. Keep updates brief and useful.
