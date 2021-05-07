#!/bin/bash
#SBATCH --array=0-10%2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --time=11:59:00

module load anaconda/3
conda activate sequoia

cd ~/Sequoia
pip install -e .[hpo,monsterkong]

sequoia --data_dir $SLURM_TMPDIR "$@"
