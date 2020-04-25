#!/bin/bash

NAME="${1:?'Name must be set'}"
OUT="$SCRATCH/slurm_out/$NAME/%x-%j.out"
ARGS=${@:2}

echo "Sweep with name '$NAME' and with args '$ARGS'"
# Create the slurm output dir if it doesn't exist already.

# activate the virtual environment (only used to download the datasets)
source ~/ENV/bin/activate
python -m scripts.download_datasets --data_dir "$SCRATCH/data"
deactivate

zip -u "$SCRATCH/data.zip" "$SCRATCH/data"

mkdir -p "$SCRATCH/slurm_out/$NAME"

sbatch --output $OUT --job-name baseline                ./scripts/sbatch_job.sh --run_name "${NAME}_baseline"       $ARGS
sbatch --output $OUT --job-name rotation_01_nc          ./scripts/sbatch_job.sh --run_name "${NAME}_rotation_01"    $ARGS --rotation.coef 0.1 --rotation.compare False
sbatch --output $OUT --job-name vae_001                 ./scripts/sbatch_job.sh --run_name "${NAME}_vae_001"        $ARGS --vae.coef 0.01
sbatch --output $OUT --job-name simclr_1     	        ./scripts/sbatch_job.sh --run_name "${NAME}_irm_001"        $ARGS --simclr.coef 1
sbatch --output $OUT --job-name rotation_01_nc_vae_001  ./scripts/sbatch_job.sh --run_name "${NAME}_rotation_01_nc_vae_001"    $ARGS --rotation.coef 0.1 --rotation.compare False --vae.coef 0.01
sbatch --output $OUT --job-name rotation_01_nc_simclr_1 ./scripts/sbatch_job.sh --run_name "${NAME}_rotation_01_nc_simclr_1"        $ARGS --rotation.coef 0.1 --rotation.compare False --simclr.coef 1
sbatch --output $OUT --job-name vae_001_simclr_1        ./scripts/sbatch_job.sh --run_name "${NAME}_vae_001_simclr_1"        $ARGS --vae.coef 0.01 --simclr.coef 1
sbatch --output $OUT --job-name rotation_01_nc_vae_001_simclr_1 ./scripts/sbatch_job.sh --run_name "${NAME}_rotation_01_nc_vae_001_simclr_1" $ARGS --rotation.coef 0.1 --rotation.compare False --vae.coef 1 --simclr.coef 1
