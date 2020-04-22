#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_gpu    # Yoshua pays for your job
#SBATCH --gres=gpu:1                    # Request GPU "generic resources"
#SBATCH --cpus-per-task=6               # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G                       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=24:00:00                 # The job will run for 24 hours max
#SBATCH --output /scratch/normandf/%x-%j.out  # Write the log in $SCRATCH

module load httpproxy
module load python/3.7

# activate a venv in the user home directory (not ideal, but needed since we can't install some needed packages in the compute node).
source ~/ENV/bin/activate

# 1. Create your environement locally
#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate
#pip install --no-index torch torchvision
#pip install simple-parsing

cd ~/repos/SSCL

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp --update $SCRATCH/data.zip -d $SLURM_TMPDIR

# 3. Eventually unzip your dataset
unzip -n $SLURM_TMPDIR/data.zip -d $SLURM_TMPDIR

# WANDB_MODE=dryrun
# export WANDB_MODE
echo "Executing main.py with additional args: ${@:1}"

python -u main.py task-incremental \
    --data_dir $SLURM_TMPDIR/data \
    --log_dir_root $SLURM_TMPDIR/SSCL "${@:1}"

rsync -r $SLURM_TMPDIR/SSCL/* $SCRATCH/SSCL
wandb sync $SCRATCH/SSCL/wandb/ # Not guaranteed to work given CC's network restrictions.
# To make sure, run `wandb sync $SCRATCH/SSCL/wandb/` from a login node

