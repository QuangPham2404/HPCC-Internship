# Containerized Multi-Node PyTorch DDP Smoke Test

This is a model setup script that you can use as a starting point and change according to your cluster, container, GPU layout, and benchmark needs. The workflow launches a synthetic ResNet-50 Distributed Data Parallel test across multiple PBS GPU nodes using Apptainer and OpenMPI.

## One-time container download

Download the NVIDIA PyTorch container from NGC before submitting the PBS script:

<https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/pytorch/->

Use the pinned `26.04` container version. Do not install the `latest` tag, because it may include package conflicts and can be unstable.

```bash
module load apptainer/1.4.1
apptainer pull pytorch_26.04.sif docker://nvcr.io/nvidia/pytorch:26.04
```

After the pull completes, use the absolute path to `pytorch_26.04.sif` as the `CONTAINER` value in the containerized PBS script.

## Files

- `test_resnet50_multiNode_DDP_env.pbs` - PBS launch script for the containerized multi-node workflow.
- `resnet50_multiNode_DDP_env.py` - example PyTorch DDP smoke-test workload.

## What users should change

Before submitting, update these values in `test_resnet50_multiNode_DDP_env.pbs`:

| Setting | Purpose |
|---|---|
| `#PBS -P <project>` | Your PBS project or account. |
| `#PBS -q <queue>` | GPU queue/partition for the job. |
| `#PBS -l select=...` | Node, CPU, GPU, and MPI process layout. |
| `walltime` | Maximum runtime for the job. |
| `module load apptainer/...` | Apptainer module available on your cluster. |
| `module load nvhpc/...` | NVHPC/HPC-X OpenMPI stack used by `mpirun`. |
| `CONTAINER=<path>` | Absolute path to the PyTorch Apptainer `.sif` image. |
| `SCRIPT=<path>` | Path to `resnet50_multiNode_DDP_env.py` or another DDP script. |
| `NCCL_IB_HCA` | Cluster-specific InfiniBand HCA list, if needed. |
| `NCCL_DEBUG` / `NCCL_DEBUG_SUBSYS` | Keep enabled for debugging or reduce for cleaner logs. |

`NPROCS`, `NNODES`, `GPUS_PER_NODE`, `MASTER_ADDR`, and `MASTER_PORT` are derived from the PBS allocation. `MASTER_PORT` is based on `PBS_JOBID` to reduce port collisions between jobs.

## How to run

Submit from the directory containing the PBS script:

```bash
qsub test_resnet50_multiNode_DDP_env.pbs
```

The script prints the job layout, selected nodes, master address/port, NCCL settings, and pass/fail status.

## Python smoke-test script

`resnet50_multiNode_DDP_env.py` is an example smoke test. It reads OpenMPI rank variables, maps one process to one GPU, initializes PyTorch DDP with `env://`, and trains ResNet-50 on synthetic data.

Useful values to edit for a smaller or larger test:

- `dataset_size`
- `image_size`
- `batch_size`
- `epochs`
- `num_workers`

`batch_size` is per GPU. The effective global batch size is `batch_size * world_size`.
