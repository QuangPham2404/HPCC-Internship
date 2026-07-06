# HPCC-Internship

Benchmarking and launch scripts developed during my HPCC internship for evaluating GPU/HPC workloads on a multi-GPU cluster environment.

This repository focuses on comparing containerized and non-containerized execution workflows across representative AI and HPC benchmarks, including:

- PyTorch distributed training
- vLLM inference workloads
- HPL
- HPCG

The scripts are intended to document practical cluster benchmarking workflows, including job submission, environment setup, distributed launch configuration, and runtime validation.

## Hardware Context

The scripts were tested on an internal multi-GPU cluster environment. Hardware details are intentionally kept high-level for confidentiality:

- Dual Intel CPU compute nodes
- 56 CPU cores per node
- 8× NVIDIA H200 NVL GPUs per node
- Multiple high-speed network interfaces per node

## Confidentiality Note

This repository does not include confidential benchmark results, internal hostnames, IP addresses, scheduler project names, credentials, or site-specific configuration values. Any sensitive cluster-specific details have been removed or generalized.

The purpose of this repository is to showcase benchmarking methodology, distributed launch workflows, and reproducible script structure rather than disclose internal system performance.


