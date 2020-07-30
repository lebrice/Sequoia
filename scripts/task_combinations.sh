#!/bin/bash

NAME="${1:?'Name must be set'}"
N_JOBS="${2:?'N_JOBS must be set'}"

OUT="$SCRATCH/slurm_out/$NAME/%x-%A_%a.out"
ARGS="${@:3}"

echo "Sweep with name '$NAME' and with args '$ARGS'"
echo "OUT pattern: $OUT"
echo "Number of jobs per task: $N_JOBS"
# Create the slurm output dir if it doesn't exist already.

if [[ $HOSTNAME == *"blg"* ]] && [[ $SETUP -eq 1 ]]; then
    echo "Downloading the datasets and models from the login node since we're on Beluga."
    source scripts/beluga/setup.sh
    export SETUP=1
fi

# activate the virtual environment (only used to download the datasets)

mkdir -p "$SCRATCH/slurm_out/$NAME"

ROT_ARGS="--rotation.coef 1 --rotation.compare False"
AE_ARGS="--ae.coef 0.01"
SIMCLR_ARGS="--simclr.coef 1"

./scripts/sbatch_job.sh baseline            $N_JOBS --run_group $NAME --run_name "baseline"            $ARGS
./scripts/sbatch_job.sh rotation            $N_JOBS --run_group $NAME --run_name "rotation"            $ARGS $ROT_ARGS
./scripts/sbatch_job.sh ae                  $N_JOBS --run_group $NAME --run_name "ae"                  $ARGS $AE_ARGS
./scripts/sbatch_job.sh simclr              $N_JOBS --run_group $NAME --run_name "simclr"              $ARGS $SIMCLR_ARGS
./scripts/sbatch_job.sh rotation_ae         $N_JOBS --run_group $NAME --run_name "rotation_ae"         $ARGS $ROT_ARGS $AE_ARGS
./scripts/sbatch_job.sh rotation_simclr     $N_JOBS --run_group $NAME --run_name "rotation_simclr"     $ARGS $ROT_ARGS $SIMCLR_ARGS
./scripts/sbatch_job.sh ae_simclr           $N_JOBS --run_group $NAME --run_name "ae_simclr"           $ARGS $AE_ARGS $SIMCLR_ARGS
./scripts/sbatch_job.sh rotation_ae_simclr  $N_JOBS --run_group $NAME --run_name "rotation_ae_simclr"  $ARGS $ROT_ARGS $AE_ARGS $SIMCLR_ARGS
