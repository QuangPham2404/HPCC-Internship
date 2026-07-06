# Bare-Metal Multi-Node PyTorch DDP Smoke Test

This is a model setup script that you can use as a starting point and change according to your cluster modules, GPU layout, network settings, and benchmark needs. The workflow launches a synthetic ResNet-50 Distributed Data Parallel test across multiple PBS GPU nodes without a container.

## Files

- `test_resnet50_multiNode_DDP_NC_env.pbs` - PBS launch script for the bare-metal multi-node workflow.
- `resnet50_multiNode_DDP_NC_env.py` - example PyTorch DDP smoke-test workload.

## What users should change

Before submitting, update these values in `test_resnet50_multiNode_DDP_NC_env.pbs`:

| Setting | Purpose |
|---|---|
| `#PBS -P <project>` | Your PBS project or account. |
| `#PBS -q <queue>` | GPU queue/partition for the job. |
| `#PBS -l select=...` | Node, CPU, GPU, and MPI process layout. |
| `walltime` | Maximum runtime for the job. |
| `module load python/...` | Python module available on your cluster. |
| `module load pytorch/...` | PyTorch module available on your cluster. |
| `module load nvhpc/...` | NVHPC/HPC-X OpenMPI stack used by `mpirun`. |
| `SCRIPT=<path>` | Path to `resnet50_multiNode_DDP_NC_env.py` or another DDP script. |
| `NCCL_IB_HCA` | Cluster-specific InfiniBand HCA list, if needed. |
| `NCCL_DEBUG` / `NCCL_DEBUG_SUBSYS` | Keep enabled for debugging or reduce for cleaner logs. |

`NPROCS`, `NNODES`, `GPUS_PER_NODE`, `MASTER_ADDR`, and `MASTER_PORT` are derived from the PBS allocation. `MASTER_PORT` is based on `PBS_JOBID` so separate jobs are less likely to reuse the default PyTorch rendezvous port.

## How to run

Submit from the directory containing the PBS script:

```bash
qsub test_resnet50_multiNode_DDP_NC_env.pbs
```

The script prints the job layout, selected nodes, master address/port, NCCL settings, and pass/fail status.

## Python smoke-test script

`resnet50_multiNode_DDP_NC_env.py` is an example smoke test. It reads OpenMPI rank variables, maps one process to one GPU, initializes PyTorch DDP with `env://`, and trains ResNet-50 on synthetic data.

Useful values to edit for a smaller or larger test:

- `dataset_size`
- `image_size`
- `batch_size`
- `epochs`
- `num_workers`

`batch_size` is per GPU. The effective global batch size is `batch_size * world_size`.
