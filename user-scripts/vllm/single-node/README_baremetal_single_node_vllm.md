# Bare-Metal Single-Node vLLM Smoke Test

This is a model script template. Replace the placeholder project, queue, path, storage, and runtime values with valid cluster-specific values before submitting it.

This workflow launches a vLLM OpenAI-compatible API server directly from the cluster's bare-metal `vllm` module, waits for the server to become ready, and sends one chat-completions request as a sanity test.

Use this script when you want to test the non-containerized vLLM stack on one GPU node. It is the bare-metal counterpart of the containerized single-node script.

## Files

- `baremetal_single_node_vllm_smoke_refined.pbs` — PBS job script for launching and testing vLLM on a single node without a container.
- `README_baremetal_single_node_vllm.md` — this guide.

## What users must edit

Before submitting the job, check these sections in the script:

1. **PBS account, queue, and resources**
   - `#PBS -P <project>
   - `#PBS -q <queue>
   - active `#PBS -l select=...ngpus=...`
   - `TP_SIZE`

   The requested GPU count and `TP_SIZE` must match.

2. **vLLM module**
   - `module load vllm/0.19.0`

   Change this only if the cluster provides another tested bare-metal vLLM module.

3. **Working directory**
   - `BENCH_ROOT="/path/to/shared/bench_root"`

   This should point to a writable persistent storage location with enough capacity for model files, caches, logs, and temporary files.

4. **Model and server settings**
   - `MODEL_ID="Qwen/Qwen2.5-7B-Instruct"`
   - `PORT=8000`
   - `HOST="0.0.0.0"`

   Change `MODEL_ID` to serve a different Hugging Face model. Change `PORT` if 8000 is already occupied. Use `HOST="127.0.0.1"` for local-only serving.

## Submit the job

From the directory containing the script:

```bash
qsub baremetal_single_node_vllm_smoke_refined.pbs
```

PBS writes the main job output to the submission directory. The script also writes vLLM logs and request/response artifacts to the configured output directories.

## Expected outputs

The script creates:

- vLLM server log under `$BENCH_ROOT/logs/`
- one run directory named like `baremetal_qwen2p5_7b_tp<...>_<...>`
- `models.json` from `/v1/models`
- `request.json` containing the smoke-test prompt
- `response.json` containing the raw API response
- `response_text.txt` containing the extracted assistant response, if `python3` is available

## Bare-metal-specific behavior

Unlike the containerized workflow, this script does not load Apptainer and does not use a `.sif` image. It relies on the cluster module environment.

The script also normalizes `CUDA_VISIBLE_DEVICES`. On some PBS systems, allocated GPUs appear as UUIDs such as `GPU-xxxx`. The bare-metal vLLM stack used here expects integer GPU IDs such as `0,1`, so the script maps UUIDs to integer indices before launching vLLM.

## Common checks

To inspect the run:

```bash
cat <pbs-output-file>
tail -n 120 $BENCH_ROOT/logs/<vllm-log-file>
cat <run-dir>/response_text.txt
```

Common failure points:

- `TP_SIZE` does not match the requested `ngpus`.
- The selected model is too large for the requested GPU count.
- `BENCH_ROOT` is not writable or has insufficient space.
- The bare-metal `vllm` module is not available on the allocated node.
- Port 8000 is already in use.

## Switching GPU count

To run on 2 GPUs, comment out the active 1-GPU PBS line and `TP_SIZE=1`, then activate the 2-GPU PBS line and `TP_SIZE=2`.

Example:

```bash
##PBS -l select=1:ncpus=16:mem=64gb:ngpus=1
#export TP_SIZE=1

#PBS -l select=1:ncpus=16:mem=96gb:ngpus=2
export TP_SIZE=2
```

## Cleanup

The script starts vLLM in the background and installs a cleanup handler. When the PBS job exits, the handler terminates the vLLM process group so that GPUs and ports are not left occupied by orphaned server processes.
