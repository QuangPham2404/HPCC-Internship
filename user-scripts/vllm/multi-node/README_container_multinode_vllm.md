# Containerized Multi-Node vLLM with Ray

This is a model script template. Replace the placeholder project, queue, path, storage, and runtime values with valid cluster-specific values before submitting it.

This README explains how to use `container_multinode_vllm_smoke_refined.pbs` to launch a vLLM OpenAI-compatible server across multiple PBS nodes using an Apptainer container and Ray.

The script is designed for a workflow where Ray is installed once into a persistent container home directory and then reused across different vLLM models and benchmark runs.

---

## One-time container download

Download the NVIDIA vLLM container from NGC before submitting the PBS script:

<https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/vllm/->

Use the pinned `26.04` container version. Do not install the `latest` tag, because it may include internal package conflicts and can be unstable.

```bash
module load apptainer/1.4.1
apptainer pull vllm_26.04-py3.sif docker://nvcr.io/nvidia/vllm:26.04
```

After the pull completes, use the absolute path to `vllm_26.04.sif` as the `SIF` value in the containerized PBS script and in the Ray installation commands below.

---

## What this script does

The script performs the following sequence:

1. Requests multiple PBS nodes with GPUs.
2. Loads Apptainer and the MPI stack needed to launch commands on allocated nodes.
3. Uses a persistent container home, `ENV_ROOT`, where Ray has already been installed.
4. Uses a shared working directory, `BENCH_ROOT`, for model caches, runtime caches, logs, and results.
5. Starts one Ray head process on the first allocated node.
6. Starts one Ray worker process on each remaining allocated node.
7. Verifies that the Ray cluster is ready.
8. Launches a multi-node vLLM server using Ray as the distributed executor.
9. Waits until the `/v1/models` endpoint is ready.
10. Sends one optional smoke-test prompt to verify that the server works.

---

## Files

Recommended naming:

```bash
container_multinode_vllm_smoke_refined.pbs
README_container_multinode_vllm.md
```

Submit the script with:

```bash
qsub container_multinode_vllm_smoke_refined.pbs
```

---

## One-time Ray installation

Ray must be installed inside the same container-home layout that the PBS script will use.

Set:

```bash
export ENV_ROOT="/path/to/persistent/container_env"
export SIF="/path/to/vllm_container.sif"
```

Then install Ray once:

```bash
apptainer exec --nv --cleanenv \
  --home "$ENV_ROOT" \
  --bind "$ENV_ROOT:$ENV_ROOT" \
  "$SIF" \
  python3 -m pip install --user "ray[default]"
```

Verify the installation:

```bash
apptainer exec --nv --cleanenv \
  --home "$ENV_ROOT" \
  --bind "$ENV_ROOT:$ENV_ROOT" \
  "$SIF" \
  bash -lc 'python3 -c "import ray; print(ray.__version__)" && ~/.local/bin/ray --version'
```

After this succeeds, the PBS script should point to the same `ENV_ROOT` and Ray binary:

```bash
export ENV_ROOT="/path/to/persistent/container_env"
export RAY_BIN="/path/to/persistent/container_env/.local/bin/ray"
```

---

## How to reuse Ray correctly

Ray is reusable across different models as long as the same container image and Python environment remain compatible.

For a new model, usually change only:

```bash
export BENCH_ROOT="/path/to/new/shared/model_or_run_root"
export MODEL_ID="new/model-id"
```

and, if needed:

```bash
export GPUS_PER_NODE=...
export TP_SIZE=...
export PP_SIZE=...
#PBS -l select=...
```

Do not reinstall Ray just because you changed model.

Reinstall Ray only when:

- the vLLM container image changes,
- the Python version inside the container changes,
- Ray is missing or corrupted,
- you intentionally want a different Ray version,
- the existing `ENV_ROOT` was deleted or moved.

A good mental model is:

```text
ENV_ROOT  = reusable container Python/Ray environment
BENCH_ROOT = per-model/per-run cache, logs, and results
```

---

## Required edits before running

### 1. PBS allocation

Enable exactly one PBS select line and make the layout variables consistent.

Example for 2 nodes with 2 GPUs per node:

```bash
#PBS -l select=2:ncpus=12:ngpus=2:mpiprocs=1
export GPUS_PER_NODE=2
export TP_SIZE=2
export PP_SIZE=2
```

In this template:

```text
TP_SIZE = GPUs used per node
PP_SIZE = number of nodes
Total model GPUs = TP_SIZE × PP_SIZE
```

`mpiprocs=1` is intentional. The script launches one Ray process per node, and Ray manages GPU work inside each node.

### 2. Container image

Set:

```bash
export SIF="/path/to/vllm_container.sif"
```

The path must be visible on all allocated nodes.

### 3. Shared working directory

Set:

```bash
export BENCH_ROOT="/path/to/shared/bench_root"
```

This must be visible from every allocated node. It stores Hugging Face cache, vLLM cache, runtime caches, Ray logs, and run outputs.

### 4. Ray environment

Set:

```bash
export ENV_ROOT="/path/to/persistent/container_env"
export RAY_BIN="/path/to/persistent/container_env/.local/bin/ray"
```

These should point to the environment where Ray was installed with `pip install --user`.

### 5. Model ID

Set the model in the vLLM launch section:

```bash
export MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
```

There is also an inner `MODEL_ID` export inside the `setsid bash -lc` block. Keep it consistent with the outer model setting.

---

## Choosing TP and PP for a model

For this template:

```text
tensor parallelism = within a node
pipeline parallelism = across nodes
```

Examples:

```text
2 nodes × 1 GPU each:
  GPUS_PER_NODE=1
  TP_SIZE=1
  PP_SIZE=2

2 nodes × 2 GPUs each:
  GPUS_PER_NODE=2
  TP_SIZE=2
  PP_SIZE=2

4 nodes × 2 GPUs each:
  GPUS_PER_NODE=2
  TP_SIZE=2
  PP_SIZE=4
```

Before changing TP/PP, confirm that the target model supports the chosen tensor-parallel size and that the model can fit across the selected total GPU count.

---

## Important cluster-specific settings

### Head node IP

The script resolves the first allocated node as the Ray head:

```bash
HEAD_NODE="${NODES[0]}"
HEAD_IP="$(getent ahostsv4 "$HEAD_NODE" | awk '{print $1; exit}')"
```

If this resolves to the wrong network interface, replace the resolved `HEAD_IP` logic with the preferred IB/IPoIB hostname or IP address for your cluster.

### NCCL settings

The current NCCL settings are tuned for the GAAS cluster:

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
```

On a different cluster, users may need to update `NCCL_IB_HCA` to match the available HCAs.

### Debug log volume

The script uses:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
```

This is useful for validation. For routine use, users can reduce log volume by setting `NCCL_DEBUG=WARN` or removing `NCCL_DEBUG_SUBSYS`.

---

## Expected outputs

At the end of a successful run, the script prints paths similar to:

```text
RUN_DIR:       .../results/container_qwen2p5_7b_multinode_tp2_pp2_<jobid>_<timestamp>
Ray status:    .../ray_status.txt
Ray logs:      .../ray_logs
vLLM log:      .../vllm_qwen2p5_7b_multinode_tp2_pp2_<jobid>.log
Response JSON: .../response.json
```

Useful files:

```text
RUN_DIR/ray_status.txt
RUN_DIR/ray_logs/ray_head_<node>.log
RUN_DIR/ray_logs/ray_worker_<node>.log
RUN_DIR/models.json
RUN_DIR/request.json
RUN_DIR/response.json
RUN_DIR/response_text.txt
VLLM_LOG_FILE
```

---

## Common checks

### Check whether Ray was installed correctly

Run the verification command from the one-time install section. If Ray is not found, check `ENV_ROOT` and `RAY_BIN`.

### Check whether every node sees GPUs

The script prints:

```bash
hostname
nvidia-smi -L
```

for each allocated node before starting Ray.

### Check Ray startup

If Ray does not become ready, inspect:

```text
RUN_DIR/ray_logs/ray_head_<head_node>.log
RUN_DIR/ray_logs/ray_worker_<worker_node>.log
RUN_DIR/ray_status_attempt_*.txt
```

### Check vLLM startup

If the API never becomes ready, inspect:

```text
VLLM_LOG_FILE
RUN_DIR/ray_status.txt
```

Common causes include:

- wrong `MODEL_ID`,
- model too large for the selected GPU layout,
- unsupported `TP_SIZE`,
- incorrect `HEAD_IP` network selection,
- wrong or missing `RAY_BIN`,
- NCCL selecting the wrong network device,
- stale Ray processes from a previous failed job.

The script already pre-cleans stale Ray at the beginning and stops Ray again during cleanup.

---

## Hosting without the smoke test

The final chat-completion request is optional. It is included to verify that the server works.

For a pure hosting script, keep the readiness check but remove or comment out the smoke-test request section after:

```bash
echo "Model list saved to:"
echo "$MODELS_JSON"
```

Make sure the PBS walltime is long enough for the intended hosting session.
