#!/bin/bash
set -euo pipefail

# ============================================================
# Start GuideLLM environment
#
# Safe to source repeatedly.
# This script does NOT delete, recreate, or reinstall the env.
#
# Usage:
#   source "$HOME/guidellm_vLLMBench/start_guidellm_env.sh"
# ============================================================

if [[ "${EUID}" -eq 0 ]]; then
  echo "ERROR: Do not source this as root."
  echo "Exit root first, then run as normal user."
  return 1 2>/dev/null || exit 1
fi

BENCH_ROOT="${BENCH_ROOT:-$HOME/guidellm_vLLMBench}"

cd "$BENCH_ROOT"

module purge
module load pytorch/2.11.0

unset PYTHONPATH
unset PYTHONHOME

if [[ ! -f "$BENCH_ROOT/guidellm_env/bin/activate" ]]; then
  echo "ERROR: guidellm_env does not exist."
  echo "Create it first with:"
  echo "  $BENCH_ROOT/reset_guidellm_env.sh"
  return 1 2>/dev/null || exit 1
fi

source "$BENCH_ROOT/guidellm_env/bin/activate"

unset PYTHONPATH
unset PYTHONHOME

echo "Activated GuideLLM environment"
echo "User:      $(whoami)"
echo "Python:    $(which python)"
python --version
echo "GuideLLM:  $(which guidellm 2>/dev/null || echo 'not installed')"
