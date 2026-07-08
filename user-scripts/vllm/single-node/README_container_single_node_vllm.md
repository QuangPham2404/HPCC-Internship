# Containerized Single-Node vLLM Smoke Test

This is a model script template. Replace the placeholder project, queue, path, storage, and runtime values with valid cluster-specific values before submitting it.

This workflow starts a vLLM OpenAI-compatible server inside an Apptainer container on one PBS GPU node, waits until the server is ready, and sends one optional smoke-test prompt.

Use this script when you want a minimal, reproducible check that:

1. the vLLM container can start on the allocated GPU node,
2. the model can be downloaded or loaded from cache,
3. tensor parallelism is configured correctly for the requested GPU count, and
4. the `/v1/chat/completions` endpoint returns a valid response.

## One-time container download

Download the NVIDIA vLLM container from NGC before submitting the PBS script:

<https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/vllm/->

Use the pinned `26.04` container version. Do not install the `latest` tag, because it may include internal package conflicts and can be unstable.

```bash
module load apptainer/1.4.1
apptainer pull vllm_26.04-py3.sif docker://nvcr.io/nvidia/vllm:26.04
```

After the pull completes, use the absolute path to `vllm_26.04.sif` as the `SIF` value in the containerized PBS script.

## Files

- `container_single_node_vllm_smoke_refined.pbs`  
  PBS job script for launching the containerized vLLM server and running the smoke test.

## What users must edit

Before submitting the job, update these values in the script.

| Setting | Purpose |
|---|---|
| `#PBS -P` | PBS project/account name. |
| `#PBS -q` | PBS queue/partition name. |
| `#PBS -l select=1:ngpus=<N>` | Number of GPUs requested on one node. |
| `TP_SIZE` | vLLM tensor-parallel size. This must match the requested GPU count for this script. |
| `SIF` | Absolute path to the Apptainer vLLM image. |
| `BENCH_ROOT` | Working directory for model cache, runtime cache, logs, and outputs. Use large project or scratch storage. |
| `MODEL_ID` | Hugging Face model identifier passed to `vllm serve`. |
| `PORT` | HTTP port for the vLLM server. Change this if the default port is already in use. |

## Recommended storage layout

The script creates the following structure under `BENCH_ROOT`:

```text
BENCH_ROOT/
├── hf_home/                 # Hugging Face model/config/tokenizer cache
├── xdg_cache/               # General runtime cache
├── vllm_cache/              # vLLM cache
├── flashinfer_cache/        # FlashInfer workspace/cache
├── torchinductor_cache/     # PyTorch Inductor cache
├── triton_cache/            # Triton cache
├── tmp/                     # Temporary files
├── logs/                    # vLLM server logs
└── results/                 # Per-run smoke-test outputs
```

Using a single `BENCH_ROOT` makes cleanup and debugging easier. Prefer a short path on large shared storage. Very long paths can sometimes cause socket or temporary-file issues in distributed/runtime components.

## How to run

Submit the script from the directory where you want the PBS output file to be created:

```bash
qsub container_single_node_vllm_smoke_refined.pbs
```

Check job status:

```bash
qstat -u "$USER"
```

After the job starts, inspect the PBS output file and the vLLM log path printed by the script.

## Expected outputs

Each run creates a unique `RUN_DIR` under:

```text
$BENCH_ROOT/results/
```

The main output files are:

| File | Description |
|---|---|
| `models.json` | Response from `/v1/models`, used to confirm server readiness. |
| `request.json` | The smoke-test request body sent to vLLM. |
| `response.json` | Raw response from `/v1/chat/completions`. |
| `response_text.txt` | Extracted assistant message text, if `python3` is available on the host. |

The vLLM server log is written to:

```text
$BENCH_ROOT/logs/
```

## Changing GPU count

For a 1-GPU run:

```bash
#PBS -l select=1:ngpus=1
export TP_SIZE=1
```

For a 2-GPU run, comment out the 1-GPU pair and enable the 2-GPU pair:

```bash
#PBS -l select=1:ngpus=2
export TP_SIZE=2
```

Only one PBS `select` line should be active. `TP_SIZE` must not exceed the number of visible GPUs.

## Changing the model

Update:

```bash
export MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
```

The model must fit within the requested GPU resources. For multi-GPU tensor parallelism, confirm that the model supports the intended tensor-parallel size.

## Common checks

If the job fails before vLLM starts:

- Confirm that the requested queue/project is valid.
- Confirm that the Apptainer module names exist on the cluster.
- Confirm that the `SIF` path exists on the compute node.

If vLLM exits before readiness:

- Inspect the last lines printed from the vLLM log.
- Check whether the model path or Hugging Face identifier is correct.
- Check whether the model requires authentication or prior login/cache setup.
- Check whether `TP_SIZE` matches the allocated GPU count.

If startup hangs for a long time:

- The first run may be downloading weights or compiling kernels.
- Inspect the vLLM log file under `$BENCH_ROOT/logs`.
- Consider using a shorter `BENCH_ROOT` or `TMPDIR` if the log mentions path/socket errors.

## Notes

The smoke-test prompt is optional. It can be edited or removed if the goal is only to start the server and keep it running for external requests.

The cleanup handler is recommended because vLLM is launched in the background. When the PBS job exits, the handler terminates the vLLM process group so that server processes do not remain behind on the node.
