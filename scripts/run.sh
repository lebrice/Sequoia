#!/bin/bash
#SBATCH --gres=gpu:1                    # Request GPU "generic resources"
#SBATCH --cpus-per-task=6               # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G                       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=24:00:00                 # The job will run for 24 hours max
#SBATCH --output /scratch/normandf/slurm_out/%x/%x-%A_%a.out  # Write stdout in $SCRATCH


cd $SCRATCH/repos/SSCL

echo "Slurm Array Job ID: $SLURM_ARRAY_TASK_ID"

source scripts/setup.sh
export TORCH_HOME="$SCRATCH/.torch"


function cleanup(){
    echo "Cleaning up and transfering files from $SLURM_TMPDIR to $SCRATCH/SSCL"
    rsync -r -u -v $SLURM_TMPDIR/SSCL/* $SCRATCH/SSCL
    if [[ $BELUGA -eq 1 ]]; then
        echo "Trying to sync just this run, since we're on Beluga.."
        wandb sync $SLURM_TMPDIR/SSCL/wandb/ # Not guaranteed to work given CC's network restrictions.
    else
        echo "Running wandb sync since we're not on Beluga"
        wandb sync $SCRATCH/SSCL
    fi
}

trap cleanup EXIT

if [[ $BELUGA -eq 1 ]]; then
    echo "Turning off wandb since we're running on Beluga."
    wandb off
else
    echo "Logging in with wandb since we're running on Cedar."
    export WANDB_API_KEY=8776db49aa9afcae1952308876d832cdbd11fcbe
    wandb login
fi

echo "Calling python -u main.py task-incremental \
    --data_dir $SLURM_TMPDIR/data \
    --log_dir_root $SLURM_TMPDIR/results \
    --run_number ${SLURM_ARRAY_TASK_ID:-0} \
    ${@}"

python -u main.py task-incremental \
    --data_dir $SLURM_TMPDIR/data \
    --log_dir_root $SLURM_TMPDIR/SSCL \
    --run_number ${SLURM_ARRAY_TASK_ID:-0} \
    ${@}

exit

