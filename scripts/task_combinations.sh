#!/bin/bash

NAME="${1:?'Name must be set'}"
OUT="$SCRATCH/slurm_out/$NAME/%x/%x-%A_%a.out"
N_JOBS="${2:?'N_JOBS must be set'}"
ARGS="${@:3}"

echo "Sweep with name '$NAME' and with args '$ARGS'"
echo "Number of jobs per task: $N_JOBS"
# Create the slurm output dir if it doesn't exist already.

if [[ $HOSTNAME == *"blg"* ]]; then
    echo "Downloading the datasets and models from the login node since we're on Beluga."
    source scripts/setup.sh

    ACCOUNT="rrg-bengioy-ad"
    PARTITION="default"  # TODO: figure out what the right partition is for this
else
    ACCOUNT=$USER
    PARTITION="long"
fi



# activate the virtual environment (only used to download the datasets)

mkdir -p "$SCRATCH/slurm_out/$NAME"

ROT_ARGS="--rotation.coef 1 --rotation.compare False"
AE_ARGS="--ae.coef 0.01"
SIMCLR_ARGS="--simclr.coef 1"

sbatch --output $OUT --job-name baseline            --account=$ACCOUNT -p=$PARTITION --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_group $NAME --run_name "baseline"            $ARGS 
sbatch --output $OUT --job-name rotation            --account=$ACCOUNT -p=$PARTITION --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_group $NAME --run_name "rotation"            $ARGS $ROT_ARGS
sbatch --output $OUT --job-name ae                  --account=$ACCOUNT -p=$PARTITION --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_group $NAME --run_name "ae"                  $ARGS $AE_ARGS
sbatch --output $OUT --job-name simclr              --account=$ACCOUNT -p=$PARTITION --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_group $NAME --run_name "simclr"              $ARGS $SIMCLR_ARGS
sbatch --output $OUT --job-name rotation_ae         --account=$ACCOUNT -p=$PARTITION --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_group $NAME --run_name "rotation_ae"         $ARGS $ROT_ARGS $AE_ARGS
sbatch --output $OUT --job-name rotation_simclr     --account=$ACCOUNT -p=$PARTITION --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_group $NAME --run_name "rotation_simclr"     $ARGS $ROT_ARGS $SIMCLR_ARGS
sbatch --output $OUT --job-name vae_simclr          --account=$ACCOUNT -p=$PARTITION --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_group $NAME --run_name "ae_simclr"           $ARGS $AE_ARGS $SIMCLR_ARGS
sbatch --output $OUT --job-name rotation_ae_simclr  --account=$ACCOUNT -p=$PARTITION --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_group $NAME --run_name "rotation_ae_simclr"  $ARGS $ROT_ARGS $AE_ARGS $SIMCLR_ARGS
