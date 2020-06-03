#!/bin/bash
#SBATCH --account=rrg-bengioy-ad    # Yoshua pays for your job
#SBATCH --gres=gpu:1                    # Request GPU "generic resources"
#SBATCH --cpus-per-task=6               # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G                       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=24:00:00                 # The job will run for 24 hours max
#SBATCH --output /scratch/normandf/slurm_out/%x/%x-%A_%a.out  # Write stdout in $SCRATCH


cd $SCRATCH/repos/SSCL

echo "Slurm Array Job ID: $SLURM_ARRAY_TASK_ID"

source scripts/setup.sh

function cleanup(){
    echo "Cleaning up and transfering files from $SLURM_TMPDIR to $SCRATCH/SSCL"
    rsync -r -u -v $SLURM_TMPDIR/SSCL/* $SCRATCH/SSCL
    wandb sync $SCRATCH/SSCL/wandb/ # Not guaranteed to work given CC's network restrictions.
}

trap cleanup EXIT

if [[ $BELUGA -eq 1 ]]; then
    echo "Turning off wandb since we're running on Beluga."
    wandb off
else
    echo "Logging in with wandb since we're running on Cedar."
    wandb login
fi

echo "Calling python -u main.py task-incremental \
    --data_dir $SLURM_TMPDIR/data \
    --log_dir_root $SLURM_TMPDIR/results \
    --run_number ${SLURM_ARRAY_TASK_ID:-0} \
    ${@}"

exec python -u main.py task-incremental \
    --data_dir $SLURM_TMPDIR/data \
    --log_dir_root $SLURM_TMPDIR/SSCL \
    --run_number ${SLURM_ARRAY_TASK_ID:-0} \
    ${@}

cleanup
exit

