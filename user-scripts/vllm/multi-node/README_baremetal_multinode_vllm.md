# Bare-Metal Multi-Node vLLM Workflow

This workflow starts a multi-node, non-containerized vLLM server under PBS and performs a single OpenAI-compatible API smoke test.

It assumes:

- `vllm/0.19.0` is available as a bare-metal module.
- `nvhpc/26.3` provides the validated HPC-X OpenMPI stack.
- Ray has already been installed once in a separate virtual environment.
- `BENCH_ROOT` is on shared storage visible from all allocated nodes.

## Files

- `baremetal_multinode_vllm_smoke_refined.pbs`  
  PBS script that launches Ray across allocated nodes, starts vLLM with Ray as the distributed backend, waits for API readiness, and sends one smoke-test prompt.

## One-time Ray setup

Ray is not installed directly into the `vllm/0.19.0` module environment. Instead, install it once in a separate virtual environment, then expose that Ray installation to the module-provided vLLM Python through `PYTHONPATH`.

The reference setup commands are included at the top of the script. The expected structure is:

```bash
module purge
module load vllm/0.19.0

export ENV_ROOT=<path>

cd "$ENV_ROOT"
python3 -m venv --system-site-packages baremetal_ray
source baremetal_ray/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --force-reinstall "ray[default]==2.56.0"
```

Verify Ray before submitting the PBS job:

```bash
python -c "import ray; print(ray.__version__)"
python -m ray.scripts.scripts --version
python -c "import google.protobuf; print('protobuf OK')"
python -c "import click, filelock, jsonschema, yaml, requests; print('core deps OK')"
```

## Ray reuse rule

Reuse the same `BAREMETAL_RAY_ENV` across different models when the loaded vLLM module, Python version, and Ray version remain compatible.

You usually do **not** need to reinstall Ray when changing:

- `BENCH_ROOT`
- model ID
- number of GPUs per node
- tensor-parallel size
- pipeline-parallel size
- output/log directory

Rebuild the Ray virtual environment when changing:

- the `vllm/0.19.0` module to another vLLM module
- the Python version behind the vLLM module
- the required Ray version
- the cluster software stack in a way that breaks Ray imports or dependencies

## Required edits before running

Update these paths for your account or project:

```bash
export BENCH_ROOT=<path>
export BAREMETAL_RAY_ENV=<path>
```

`BENCH_ROOT` should be model/run storage on shared storage. `BAREMETAL_RAY_ENV` should point to the reusable Ray virtual environment.

## Selecting the node/GPU layout

Enable exactly one PBS select block. The default script uses:

```bash
#PBS -l select=2:ncpus=12:ngpus=1:mpiprocs=1
export GPUS_PER_NODE=1
export TP_SIZE=1
export PP_SIZE=2
```

The convention is:

- `GPUS_PER_NODE`: GPUs allocated per node.
- `TP_SIZE`: tensor parallelism within each node.
- `PP_SIZE`: pipeline parallelism across nodes.

For example:

| Layout | PBS shape | TP_SIZE | PP_SIZE |
|---|---:|---:|---:|
| 2 nodes x 1 GPU | `select=2:ngpus=1:mpiprocs=1` | 1 | 2 |
| 2 nodes x 2 GPUs | `select=2:ngpus=2:mpiprocs=1` | 2 | 2 |

Keep `mpiprocs=1`. Ray starts one head or worker process per node, and vLLM/Ray manages GPU placement internally.

## Changing the model

The current script launches:

```bash
Qwen/Qwen2.5-7B-Instruct
```

When changing models, update both places:

1. The `vllm serve` command.
2. The smoke-test JSON `"model"` field.

Also update `RUN_TAG` and `VLLM_LOG_FILE` names if you want logs to reflect the new model name.

Before changing TP/PP sizes, confirm that the model can run with the requested tensor/pipeline parallel layout.

## How the script works

The script:

1. Loads the bare-metal vLLM and NVHPC/HPC-X modules.
2. Normalizes `CUDA_VISIBLE_DEVICES` to integer GPU IDs.
3. Adds the Ray virtual environment site-packages path to `PYTHONPATH`.
4. Derives the allocated PBS nodes from `PBS_NODEFILE`.
5. Uses the first allocated node as the Ray head.
6. Starts one Ray worker on each remaining node.
7. Checks Ray cluster status with retries.
8. Launches vLLM on the Ray head with `--distributed-executor-backend ray`.
9. Waits for `/v1/models` to become ready.
10. Sends one optional smoke-test prompt.

## Submitting the job

Submit from the directory containing the script:

```bash
qsub baremetal_multinode_vllm_smoke_refined.pbs
```

The script changes back to `PBS_O_WORKDIR`, so relative job outputs are anchored to the submission directory where applicable.

## Expected outputs

After a successful run, check the final summary printed in the PBS log:

- `RUN_DIR`
- `Ray status`
- `Ray logs`
- `vLLM log`
- `Response JSON`

Important files include:

```bash
$RUN_DIR/ray_status.txt
$RUN_DIR/ray_logs/
$RUN_DIR/models.json
$RUN_DIR/request.json
$RUN_DIR/response.json
$RUN_DIR/response_text.txt
$VLLM_LOG_FILE
```

## Debug checklist

If Ray does not become ready:

- Check `$RUN_DIR/ray_logs/ray_head_<head-node>.log`.
- Check each `$RUN_DIR/ray_logs/ray_worker_<node>.log`.
- Confirm all allocated nodes can import Ray and vLLM from the module Python.
- Confirm the Ray venv path in `BAREMETAL_RAY_ENV` is correct.
- Confirm `RAY_SITE_PACKAGES` matches the Python version used by the venv.

If vLLM hangs during startup:

- Check `$RUN_DIR/ray_status.txt`.
- Confirm Ray reports the expected active node count and total GPU count.
- Confirm `TP_SIZE * PP_SIZE` matches the intended total model GPU count.
- Confirm the model supports the requested parallel layout.

If vLLM fails with CUDA device parsing errors:

- Check the printed `CUDA_VISIBLE_DEVICES`.
- Confirm `GPUS_PER_NODE` is one of the supported values in the script: 1, 2, 4, or 8.

If distributed communication fails:

- Check the NCCL section.
- The current script is tuned for the validated InfiniBand/RDMA setup.
- Do not force Ethernet unless porting to a different cluster environment.

## What not to change casually

Avoid modifying these unless you are porting the workflow to another cluster:

- `run_on_node()` and its `mpirun` / `pbs_tmrsh` structure.
- `RAY_CMD` construction.
- `PYTHONPATH` linkage between the Ray venv and module Python.
- NCCL HCA selection.
- Ray head/worker launch order.
- `--distributed-executor-backend ray`.
