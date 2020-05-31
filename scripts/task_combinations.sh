#!/bin/bash

NAME="${1:?'Name must be set'}"
OUT="$SCRATCH/slurm_out/$NAME/%x-%j.out"
N_JOBS="${2:?'N_JOBS must be set'}"
ARGS=${@:3}

echo "Sweep with name '$NAME' and with args '$ARGS'"
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

sbatch --output $OUT --job-name baseline            ./scripts/sbatch_job.sh --run_name "${NAME}_baseline"                    $N_JOBS $ARGS 
sbatch --output $OUT --job-name rotation            ./scripts/sbatch_job.sh --run_name "${NAME}_rotation_1_nc"               $N_JOBS $ARGS $ROT_ARGS
sbatch --output $OUT --job-name ae                  ./scripts/sbatch_job.sh --run_name "${NAME}_ae_001"                      $N_JOBS $ARGS $AE_ARGS
sbatch --output $OUT --job-name simclr     	        ./scripts/sbatch_job.sh --run_name "${NAME}_simclr_1"                    $N_JOBS $ARGS $SIMCLR_ARGS
sbatch --output $OUT --job-name rotation_ae         ./scripts/sbatch_job.sh --run_name "${NAME}_rotation_1_nc_ae_001"        $N_JOBS $ARGS $ROT_ARGS $AE_ARGS
sbatch --output $OUT --job-name rotation_simclr     ./scripts/sbatch_job.sh --run_name "${NAME}_rotation_1_nc_simclr_1"      $N_JOBS $ARGS $ROT_ARGS $SIMCLR_ARGS
sbatch --output $OUT --job-name vae_simclr          ./scripts/sbatch_job.sh --run_name "${NAME}_ae_001_simclr_1"             $N_JOBS $ARGS $AE_ARGS $SIMCLR_ARGS
sbatch --output $OUT --job-name rotation_ae_simclr  ./scripts/sbatch_job.sh --run_name "${NAME}_rotation_01_nc_ae_001_simclr_1" $N_JOBS $ARGS $ROT_ARGS $AE_ARGS $SIMCLR_ARGS
