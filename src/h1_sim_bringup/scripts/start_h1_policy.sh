#!/usr/bin/env bash

set -e

CONDA_HOME="/home/loq/miniconda3"
CONDA_ENV="unitree-rl"
RL_REPOSITORY="/home/loq/unitree_rl_gym"
POLICY_CONFIG="h1.yaml"

if [ ! -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]; then
    echo "Error: Conda was not found at ${CONDA_HOME}"
    exit 1
fi

if [ ! -d "${RL_REPOSITORY}" ]; then
    echo "Error: Unitree RL repository was not found at ${RL_REPOSITORY}"
    exit 1
fi

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${RL_REPOSITORY}"

echo "Starting Unitree H1 MuJoCo policy..."
echo "Conda environment: ${CONDA_ENV}"
echo "Repository: ${RL_REPOSITORY}"
echo "Configuration: ${POLICY_CONFIG}"

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python \
    "${SCRIPT_DIRECTORY}/deploy_h1_ros.py" \
    "${POLICY_CONFIG}"
