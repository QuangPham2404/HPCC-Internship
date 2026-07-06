# vLLM GuideLLM Benchmark Workflows

This directory stores the scripts used to benchmark containerized and bare-metal workflows for hosting a model with vLLM and measuring it with GuideLLM. The scripts are intended as reproducible starting points: update the account, queue, storage paths, model, GPU layout, and module/container versions to match your own cluster.

## Experiment Setup

| Area | Containerized workflow | Bare-metal workflow |
|---|---|---|
| Serving stack | vLLM inside an Apptainer `.sif` image | Cluster-provided `vllm/0.19.0` module |
| Benchmark client | GuideLLM virtual environment on the host | GuideLLM virtual environment on the host |
| Model | `Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` |
| Single-node layout | 1 node, selectable 1/2/4 GPUs, `TP_SIZE` matches `ngpus` | 1 node, selectable 1/2/4 GPUs, `TP_SIZE` matches `ngpus` |
| Multi-node layout | Ray cluster, `TP_SIZE=GPUs per node`, `PP_SIZE=number of nodes` | Ray cluster, `TP_SIZE=GPUs per node`, `PP_SIZE=number of nodes` |
| Benchmark load | Single-node: constant rate `150`, duration `300s` | Single-node: constant rate `150`, duration `300s` |
| Multi-node load | Constant rate `3`, duration `300s` | Constant rate `3`, duration `300s` |
| Data shape | Synthetic text, `prompt_tokens=512`, `output_tokens=256` | Synthetic text, `prompt_tokens=512`, `output_tokens=256` |

Each benchmark starts a vLLM OpenAI-compatible server, waits for `/v1/models`, sends one smoke-test chat request, then runs `guidellm benchmark`.

## Directory Structure

```text
vLLM/
├── README.md
├── reset_guidellm_env.sh
├── start_guidellm_env.sh
├── Containerized/
│   ├── benchmark_model.pbs
│   └── benchmark_multinode_fixedRay.pbs
└── Baremetal/
    ├── benchmark_model_baremetal.pbs
    └── benchmark_multinode_baremetal_fixedRay.pbs
```

## File Purposes

`reset_guidellm_env.sh` recreates the GuideLLM Python virtual environment and installs `guidellm[recommended]`. Use it only when you intentionally want to delete and rebuild the environment.

`start_guidellm_env.sh` activates the existing GuideLLM environment. The benchmark scripts source this before running the GuideLLM client.

`Containerized/benchmark_model.pbs` runs a single-node containerized vLLM benchmark. It loads Apptainer modules, starts vLLM from the container image, and benchmarks the local API endpoint.

`Containerized/benchmark_multinode_fixedRay.pbs` runs a multi-node containerized vLLM benchmark. It starts a Ray head and workers across PBS nodes, launches vLLM with Ray, and runs GuideLLM against the head-node endpoint.

`Baremetal/benchmark_model_baremetal.pbs` runs a single-node bare-metal benchmark using the cluster vLLM module. It also normalizes `CUDA_VISIBLE_DEVICES` when PBS exposes GPU UUIDs.

`Baremetal/benchmark_multinode_baremetal_fixedRay.pbs` runs a multi-node bare-metal benchmark using the cluster vLLM module plus a separate Ray environment.

## Recreate the Benchmark

1. Copy or adapt these scripts into your working benchmark directory.

2. Create the GuideLLM client environment:

```bash
bash reset_guidellm_env.sh
source start_guidellm_env.sh
guidellm --version
```

3. Edit the PBS script you want to run. At minimum, update:

- `#PBS -P` and `#PBS -q`
- active `#PBS -l select=...`
- `TP_SIZE`, `PP_SIZE`, and `GPUS_PER_NODE` for multi-node jobs
- `SIF` for containerized jobs
- `BENCH_ROOT` for shared cache, logs, and results
- `RAY_BIN` or `BAREMETAL_RAY_ENV` for multi-node Ray jobs
- `NCCL_IB_HCA` and network settings for your cluster
- `RATE`, `DURATION`, and `--data` if changing benchmark load
- model ID in the `vllm serve` command and smoke-test request body

4. Submit the desired job:

```bash
qsub Containerized/benchmark_model.pbs
qsub Containerized/benchmark_multinode_fixedRay.pbs
qsub Baremetal/benchmark_model_baremetal.pbs
qsub Baremetal/benchmark_multinode_baremetal_fixedRay.pbs
```

5. Inspect the generated `RUN_DIR` and vLLM log. Typical outputs include:

- `models.json`
- `request.json`
- `response.json`
- `response_text.txt`
- `guidellm_benchmark.log` or `guidellm_rate*_*.log`
- Ray status and Ray logs for multi-node jobs

## Notes

The scripts currently assume PBS, NVIDIA GPUs, `nvidia-smi`, and an OpenAI-compatible vLLM endpoint. Multi-node scripts require shared storage visible from every allocated node and working Ray communication between nodes.
