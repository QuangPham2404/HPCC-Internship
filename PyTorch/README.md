# PyTorch DDP Benchmark Workflows

This directory stores the scripts used to benchmark containerized and bare-metal PyTorch Distributed Data Parallel workflows. The scripts are intended as reproducible starting points: update the account, queue, container path, module versions, GPU layout, network settings, and benchmark parameters to match your own cluster.

## Experiment Setup

| Area | Containerized workflow | Bare-metal workflow |
|---|---|---|
| Runtime stack | PyTorch inside an Apptainer `.sif` image | Cluster-provided Python and PyTorch modules |
| Model | Synthetic ResNet-50 | Synthetic ResNet-50 |
| Dataset | Synthetic image dataset | Synthetic image dataset |
| Image shape | `3 x 224 x 224` | `3 x 224 x 224` |
| Dataset size | `2,000,000` generated samples | `2,000,000` generated samples |
| Batch size | `1024` per GPU | `1024` per GPU |
| Epochs | `20` | `20` |
| Single-node launch | `torchrun --nproc_per_node=<GPU count>` | `torchrun --nproc_per_node=<GPU count>` |
| Multi-node launch | `mpirun` plus Apptainer, one process per GPU | `mpirun`, one process per GPU |
| Distributed backend | PyTorch DDP with NCCL | PyTorch DDP with NCCL |

Each benchmark prints runtime information, initializes DDP, trains ResNet-50 on synthetic data, evaluates accuracy on synthetic test data, and reports timing/throughput metrics from rank 0.

## Directory Structure

```text
PyTorch/
|-- README.md
|-- Containerized/
|   |-- singlenode-launch.pbs
|   |-- multinode-launch.pbs
|   |-- singlenode.py
|   `-- multinode.py
`-- Baremetal/
    |-- singlenode-launch.pbs
    |-- multinode-launch.pbs
    |-- singlenode.py
    `-- multinode.py
```

## File Purposes

`Containerized/singlenode-launch.pbs` submits a one-node PyTorch DDP job inside an Apptainer container. It loads Apptainer-related modules, checks container GPU visibility, and launches the Python script with `torchrun`.

`Containerized/multinode-launch.pbs` submits a multi-node containerized DDP job. It derives the MPI process count and master address from the PBS allocation, passes the needed environment variables into Apptainer, and launches one DDP process per GPU with `mpirun`.

`Baremetal/singlenode-launch.pbs` submits a one-node DDP job using cluster Python/PyTorch modules. It derives a per-job `MASTER_PORT` from `PBS_JOBID` to reduce rendezvous port collisions.

`Baremetal/multinode-launch.pbs` submits a multi-node DDP job using cluster modules and OpenMPI. It exports `MASTER_ADDR`, `MASTER_PORT`, NCCL settings, and rank information through `mpirun`.

The `.py` files are example smoke-test workloads. The single-node scripts read `torchrun` variables such as `LOCAL_RANK`, `RANK`, and `WORLD_SIZE`. The multi-node scripts read OpenMPI variables such as `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_LOCAL_RANK`, and `OMPI_COMM_WORLD_SIZE`.

## Recreate the Benchmark

1. Copy or adapt these scripts into your working benchmark directory.

2. Edit the PBS script you want to run. At minimum, update:

- `#PBS -P` and `#PBS -q`
- active `#PBS -l select=...`
- `walltime`
- container path for containerized jobs
- Python script path or script name
- module versions for bare-metal jobs
- `--nproc_per_node` for single-node jobs
- `NCCL_IB_HCA` and network settings for multi-node jobs
- benchmark parameters inside the Python script, if changing workload size

3. For single-node jobs, make the requested GPU count match `torchrun`:

```bash
#PBS -l select=1:ngpus=4
torchrun --nproc_per_node=4 ...
```

4. For multi-node jobs, make the PBS process layout match the intended one-process-per-GPU layout:

```bash
# Example: 2 nodes x 4 GPUs each
#PBS -l select=2:ncpus=48:ngpus=4:mpiprocs=4
```

5. Submit the desired job:

```bash
qsub Containerized/singlenode-launch.pbs
qsub Containerized/multinode-launch.pbs
qsub Baremetal/singlenode-launch.pbs
qsub Baremetal/multinode-launch.pbs
```

6. Inspect the PBS output file. Useful checks include:

- visible GPU count
- PyTorch and CUDA versions
- `WORLD_SIZE` or total rank count
- `MASTER_ADDR` and `MASTER_PORT` for distributed jobs
- NCCL initialization messages
- epoch timing and throughput output
- final pass/fail message from multi-node launch scripts

## Notes

The scripts assume PBS, NVIDIA GPUs, `nvidia-smi`, PyTorch with CUDA/NCCL support, and a cluster MPI stack that can launch across allocated PBS nodes. Multi-node scripts currently include cluster-specific NCCL HCA examples; update those values for your own network.
