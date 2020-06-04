#!/bin/bash

NAME="${1:?'Name must be set'}"
OUT="$SCRATCH/slurm_out/$NAME/%x/%x-%A_%a.out"
N_JOBS="${2:?'N_JOBS must be set'}"
ARGS="${@:3}"

echo "Sweep with name '$NAME' and with args '$ARGS'"
echo "Number of jobs per task: $N_JOBS"
# Create the slurm output dir if it doesn't exist already.

# activate the virtual environment (only used to download the datasets)
source ~/ENV/bin/activate
python -m scripts.download_datasets --data_dir "$SCRATCH/data"
python -m scripts.download_pretrained_models # --save_dir "$SCRATCH/checkpoints"
deactivate

zip -u "$SCRATCH/data.zip" "$SCRATCH/data"

mkdir -p "$SCRATCH/slurm_out/$NAME"

ROT_ARGS="--rotation.coef 1 --rotation.compare False"
AE_ARGS="--ae.coef 0.01"
SIMCLR_ARGS="--simclr.coef 1"

sbatch --output $OUT --job-name baseline            --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "${NAME}_baseline"            $ARGS 
sbatch --output $OUT --job-name rotation            --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "${NAME}_rotation"            $ARGS $ROT_ARGS
sbatch --output $OUT --job-name ae                  --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "${NAME}_ae"                  $ARGS $AE_ARGS
sbatch --output $OUT --job-name simclr              --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "${NAME}_simclr"              $ARGS $SIMCLR_ARGS
sbatch --output $OUT --job-name rotation_ae         --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "${NAME}_rotation_ae"         $ARGS $ROT_ARGS $AE_ARGS
sbatch --output $OUT --job-name rotation_simclr     --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "${NAME}_rotation_simclr"     $ARGS $ROT_ARGS $SIMCLR_ARGS
sbatch --output $OUT --job-name vae_simclr          --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "${NAME}_ae_simclr"           $ARGS $AE_ARGS $SIMCLR_ARGS
sbatch --output $OUT --job-name rotation_ae_simclr  --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "${NAME}_rotation_ae_simclr"  $ARGS $ROT_ARGS $AE_ARGS $SIMCLR_ARGS
