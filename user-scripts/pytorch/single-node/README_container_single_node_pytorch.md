# Containerized Single-Node PyTorch DDP Smoke Test

This is a model script template. Replace the placeholder project, queue, path, storage, and runtime values with valid cluster-specific values before submitting it.

This workflow runs a synthetic ResNet-50 Distributed Data Parallel benchmark on one PBS GPU node using an Apptainer container. It is intended as a plug-and-play smoke test for checking that the container, CUDA, PyTorch, and `torchrun` can see and use the allocated GPUs.

## One-time container download

Download the NVIDIA PyTorch container from NGC before submitting the PBS script:

<https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/pytorch/->

Use the pinned `26.04` container version. Do not install the `latest` tag, because it may include package conflicts and can be unstable.

```bash
module load apptainer/1.4.1
apptainer pull pytorch_26.04-py3.sif docker://nvcr.io/nvidia/pytorch:26.04-py3
```

After the pull completes, use the absolute path to `pytorch_26.04.sif` as the `container_path` value in the containerized PBS script.

## Files

- `test_resnet50_singleNode_DDP.pbs` - PBS launch script for the containerized workflow.
- `resnet50_singleNode_DDP.py` - example PyTorch DDP benchmark using synthetic image data.

## What users should change

Before submitting, update these values in `test_resnet50_singleNode_DDP.pbs`:

| Setting | Purpose |
|---|---|
| `#PBS -P <project>` | Your PBS project or account. |
| `#PBS -q <queue>` | The GPU queue/partition to submit to. |
| `#PBS -l select=1:ngpus=<N>` | Number of GPUs requested on one node. |
| `container_path="/path/to/pytorch_26.04.sif"` | Absolute path to the Apptainer PyTorch `.sif` image. |
| `python_script="/path/to/resnet50_singleNode_DDP.py"` | Path to `resnet50_singleNode_DDP.py` or another DDP script inside the job working directory. |
| `--nproc_per_node=<N>` | Number of DDP worker processes. This should match the requested GPU count. |
| `walltime` | Maximum runtime for the PBS job. |

The script derives `MASTER_PORT` from `PBS_JOBID` to reduce port collisions on shared compute nodes.

## How to run

Submit from the directory containing the PBS script:

```bash
qsub test_resnet50_singleNode_DDP.pbs
```

Inspect the PBS output file after the job starts. The script prints job metadata, GPU visibility, PyTorch/CUDA versions, and benchmark progress.

## Python smoke-test script

`resnet50_singleNode_DDP.py` is an example workload, not a required application. Edit it if you want to change smoke-test size or behavior:

- `dataset_size`
- `image_size`
- `batch_size`
- `epochs`
- `num_workers`

Keep `batch_size` in mind as a per-GPU value. The effective batch size is `batch_size * WORLD_SIZE`.
