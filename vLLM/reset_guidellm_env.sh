#!/bin/bash
set -euo pipefail

# ============================================================
# Reset GuideLLM Python environment
#
# Use this ONLY when you intentionally want to delete and recreate
# the GuideLLM environment.
#
# Do NOT run as root.
# ============================================================

if [[ "${EUID}" -eq 0 ]]; then
  echo "ERROR: Do not run this script as root."
  echo "Run it as your normal user: pham0094"
  exit 1
fi

cd ~/guidellm_vLLMBench

echo "User: $(whoami)"
echo "Working directory: $(pwd)"

module purge
module load pytorch/2.11.0

PY312="$(which python)"

echo "Python used to create venv: $PY312"
"$PY312" --version

unset PYTHONPATH
unset PYTHONHOME

echo "Removing old guidellm_env..."
rm -rf guidellm_env

echo "Creating fresh guidellm_env..."
"$PY312" -m venv guidellm_env

source ~/guidellm_vLLMBench/guidellm_env/bin/activate

unset PYTHONPATH
unset PYTHONHOME

echo "Activated env:"
echo "Python: $(which python)"
python --version

echo "Checking sys.path:"
python -c "import sys; print(sys.executable); print('\n'.join(sys.path))"

echo "Upgrading packaging tools..."
python -m pip install --upgrade pip "setuptools<82" wheel

echo "Installing GuideLLM..."
python -m pip install "guidellm[recommended]"

echo "Verifying GuideLLM..."
which guidellm
guidellm --help
guidellm --version || true

echo "============================================================"
echo "GuideLLM environment reset and installation complete."
echo "Env path: ~/guidellm_vLLMBench/guidellm_env"
echo "============================================================"
