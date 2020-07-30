#!/bin/bash
#SBATCH -p long
#SBATCH --gres=gpu:1                    # Request GPU "generic resources"
#SBATCH --cpus-per-task=6               # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G                       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=24:00:00                 # The job will run for 24 hours max
#SBATCH --output /network/home/normandf/slurm_out/%x/%x-%A_%a.out  # Write stdout in $SCRATCH

# Set $SCRATCH to the value of $HOME if it isn't set (does nothing except on mila cluster).
export SCRATCH=${SCRATCH:=$HOME}
export TORCH_HOME="$SCRATCH/.torch"

echo "SCRATCH : $SCRATCH SLURM_TMPDIR: $SLURM_TMPDIR TORCH_HOME: $TORCH_HOME"

cd $SCRATCH/repos/SSCL

echo "Slurm Array Job ID: $SLURM_ARRAY_TASK_ID"

source scripts/mila/setup.sh

function cleanup(){
    echo "Cleaning up and transfering files from $SLURM_TMPDIR to $SCRATCH/SSCL"
    rsync -r -u -v $SLURM_TMPDIR/SSCL/* $SCRATCH/SSCL
    echo "No need to run wandb sync since we're not on Beluga"
    # wandb sync $SLURM_TMPDIR/SSCL
}

trap cleanup EXIT

echo "Logging in with wandb."
# export WANDB_API_KEY="8776db49aa9afcae1952308876d832cdbd11fcbe"
wandb login 8776db49aa9afcae1952308876d832cdbd11fcbe

echo "Calling python -u task_incremental.py \
    --data_dir $SLURM_TMPDIR/data \
    --log_dir_root $SLURM_TMPDIR/results \
    --run_number ${SLURM_ARRAY_TASK_ID:-0} \
    ${@}"

python -u task_incremental.py \
    --data_dir $SLURM_TMPDIR/data \
    --log_dir_root $SLURM_TMPDIR/SSCL \
    --run_number ${SLURM_ARRAY_TASK_ID:-0} \
    ${@}

exit

