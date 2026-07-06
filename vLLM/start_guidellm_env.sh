#!/bin/bash
set -euo pipefail

# ============================================================
# Start GuideLLM environment
#
# Safe to source repeatedly.
# This script does NOT delete, recreate, or reinstall the env.
#
# Usage:
#   source ~/guidellm_vLLMBench/start_guidellm_env.sh
# ============================================================

if [[ "${EUID}" -eq 0 ]]; then
  echo "ERROR: Do not source this as root."
  echo "Exit root first, then run as normal user."
  return 1 2>/dev/null || exit 1
fi

cd ~/guidellm_vLLMBench

module purge
module load pytorch/2.11.0

unset PYTHONPATH
unset PYTHONHOME

if [[ ! -f ~/guidellm_vLLMBench/guidellm_env/bin/activate ]]; then
  echo "ERROR: guidellm_env does not exist."
  echo "Create it first with:"
  echo "  ~/guidellm_vLLMBench/reset_guidellm_env.sh"
  return 1 2>/dev/null || exit 1
fi

source ~/guidellm_vLLMBench/guidellm_env/bin/activate

unset PYTHONPATH
unset PYTHONHOME

echo "Activated GuideLLM environment"
echo "User:      $(whoami)"
echo "Python:    $(which python)"
python --version
echo "GuideLLM:  $(which guidellm 2>/dev/null || echo 'not installed')"
