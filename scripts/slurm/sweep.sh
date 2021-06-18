#!/bin/bash
#SBATCH --array=0-10%2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --time=11:59:00
set -o errexit    # Used to exit upon error, avoiding cascading errors
set -o errtrace    # Show error trace
set -o pipefail   # Unveils hidden failures

module load anaconda/3
conda activate sequoia
cd ~/Sequoia

# TODO: Set data_dir in Config to `DATA_DIR` as a priority, and then as SLURM_TMPDIR/DATA (not just SLURM_TMPDIR!)
cp -r data $SLURM_TMPDIR/

export DATA_DIR=$SLURM_TMPDIR/data

#pip install -e .[hpo,monsterkong,avalanche]


# TODO: Change the setting, the number of tasks, the method, etc.
/home/mila/n/normandf/.conda/envs/sequoia/bin/sequoia_sweep --data_dir $SLURM_TMPDIR/data "$@"
