#!/bin/bash

NAME="${1:?'Name must be set'}"
OUT="$SCRATCH/slurm_out/$NAME/%x-%A_%a.out"
N_JOBS="${2:?'N_JOBS must be set'}"
ARGS="${@:3}"
#ARGS="--multihead --unsupervised_epochs_per_task 0 \
#     --supervised_epochs_per_task 50 --no_wandb_cleanup  \
#     --tags cifar100 debugging ewc resnet18 --run_group ewc_sweep
#     --encoder_model resnet18 --pretrained \
#     --dataset cifar100 --n_classes_per_task 20 \
#     "

echo "Sweep with name '$NAME' and with args '$ARGS'"
echo "Number of jobs per task: $N_JOBS"
# Create the slurm output dir if it doesn't exist already.

if [[ $HOSTNAME == *"blg"* ]] && [[ $SETUP -eq 1 ]]; then
    # activate the virtual environment (only used to download the datasets, if we are on beluga)
    echo "Downloading the datasets and models from the login node since we're on Beluga."
    source scripts/beluga/setup.sh
    export SETUP=1
fi

zip -u "$SCRATCH/data.zip" "$SCRATCH/data"

mkdir -p "$SCRATCH/slurm_out/$NAME"

./scripts/sbatch_job.sh baseline   $N_JOBS --run_name "baseline"   $ARGS
./scripts/sbatch_job.sh ewc_01     $N_JOBS --run_name "ewc_01"     $ARGS --ewc.coef 0.1
./scripts/sbatch_job.sh ewc_1      $N_JOBS --run_name "ewc_1"      $ARGS --ewc.coef 1
./scripts/sbatch_job.sh ewc_10     $N_JOBS --run_name "ewc_10"     $ARGS --ewc.coef 10
./scripts/sbatch_job.sh ewc_100    $N_JOBS --run_name "ewc_100"    $ARGS --ewc.coef 100
./scripts/sbatch_job.sh ewc_1000   $N_JOBS --run_name "ewc_1000"   $ARGS --ewc.coef 1000
./scripts/sbatch_job.sh ewc_10000  $N_JOBS --run_name "ewc_10000"  $ARGS --ewc.coef 10000
./scripts/sbatch_job.sh ewc_100000 $N_JOBS --run_name "ewc_100000" $ARGS --ewc.coef 100000
