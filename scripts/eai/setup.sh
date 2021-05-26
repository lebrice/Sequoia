#!/bin/bash
set -o errexit    # Used to exit upon error, avoiding cascading errors
set -o errtrace    # Show error trace
set -o pipefail   # Unveils hidden failures
# set -o nounset    # Exposes unset variables

# Script that does all the 'setup' for either the `sequoia` or `sequoia_sweep` commands
# to run correctly on the eai cluster.

source /opt/conda/bin/activate

# NOTE: ~ is normally /tmp/
# Create a new conda environment:
mkdir -p ~/.conda/envs
conda create -y --prefix ~/.conda/envs/sequoia_temp
conda activate ~/.conda/envs/sequoia_temp

cp -r /home/toolkit/.mujoco ~/

cd /workspace/Sequoia

pip install -e .[all]
sequoia --help
# sequoia --data_dir $SLURM_TMPDIR "$@"
