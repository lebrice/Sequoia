#!/bin/bash
#SBATCH --account=rrg-bengioy-ad        # Yoshua pays for your job
#SBATCH --gres=gpu:1                    # Request GPU "generic resources"
#SBATCH --cpus-per-task=6               # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G                       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=48:00:00                 # The job will run for 24 hours max
#SBATCH --output /network/home/normandf/slurm_out/%x-%A_%a.out  # Write stdout in $SCRATCH

export TORCH_HOME="$SCRATCH/.torch"
echo "SCRATCH : $SCRATCH SLURM_TMPDIR: $SLURM_TMPDIR TORCH_HOME: $TORCH_HOME"
cd $SCRATCH/repos/SSCL

echo "Slurm Array Job ID: $SLURM_ARRAY_TASK_ID"

source ~/ENVS/SSCl/bin/activate
export WANDB_DIR=$SCRATCH/SSCL/wandb

function cleanup(){
    echo "Cleaning up and transfering files from $SLURM_TMPDIR to $SCRATCH/SSCL"
    rsync -r -u -v $SLURM_TMPDIR/SSCL/* $SCRATCH/SSCL
    echo "Trying to sync just this run, since we're on Beluga.."
    wandb sync $SLURM_TMPDIR/SSCL/ # Not guaranteed to work given CC's network restrictions.
}

trap cleanup EXIT

echo "Turning off wandb since we're running on Beluga."
wandb off

echo "Calling python -u main.py task-incremental \
    --data_dir $SLURM_TMPDIR/data \
    --log_dir_root $SCRATCH/SSCL/results \
    --run_number ${SLURM_ARRAY_TASK_ID:-0} \
    ${@}"

python -u task_incremental.py \
    --data_dir $SLURM_TMPDIR/data \
    --log_dir_root $SLURM_TMPDIR/SSCL/results \
    --run_number ${SLURM_ARRAY_TASK_ID:-0} \
    ${@}

exit

