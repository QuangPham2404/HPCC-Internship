# Bare-Metal Single-Node PyTorch DDP Smoke Test

This is a model script template. Replace the placeholder project, queue, path, storage, and runtime values with valid cluster-specific values before submitting it.

This workflow runs the same synthetic ResNet-50 Distributed Data Parallel benchmark directly from cluster modules, without an Apptainer container. Use it to verify that the bare-metal Python/PyTorch module stack can launch `torchrun` and use the allocated GPUs.

## Files

- `test_resnet50_singleNode_DDP_noCon.pbs` - PBS launch script for the bare-metal workflow.
- `resnet50_singleNode_DDP_noCon.py` - example PyTorch DDP benchmark using synthetic image data.

## What users should change

Before submitting, update these values in `test_resnet50_singleNode_DDP_noCon.pbs`:

| Setting | Purpose |
|---|---|
| `#PBS -P` | Your PBS project or account. |
| `#PBS -q` | The GPU queue/partition to submit to. |
| `#PBS -l select=1:ngpus=<N>` | Number of GPUs requested on one node. |
| `module load python/...` | Python module available on your cluster. |
| `module load pytorch/...` | PyTorch module available on your cluster. |
| `--nproc_per_node=<N>` | Number of DDP worker processes. This should match the requested GPU count. |
| `walltime` | Maximum runtime for the PBS job. |

The script derives `MASTER_PORT` from `PBS_JOBID` and passes it to `torchrun`. Even on one node, DDP needs a rendezvous port. Using a per-job port helps avoid clashes with other jobs or stale processes using the default PyTorch port.

## How to run

Submit from the directory containing the PBS script:

```bash
qsub test_resnet50_singleNode_DDP_noCon.pbs
```

Inspect the PBS output file after the job starts. The script prints job metadata, GPU visibility, Python/PyTorch versions, and benchmark progress.

## Python smoke-test script

`resnet50_singleNode_DDP_noCon.py` is an example workload for validation. Edit it if you want a smaller or larger smoke test:

- `dataset_size`
- `image_size`
- `batch_size`
- `epochs`
- `num_workers`

Keep `batch_size` in mind as a per-GPU value. The effective batch size is `batch_size * WORLD_SIZE`.
