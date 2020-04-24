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


sbatch --output $OUT --job-name baseline      		./scripts/sbatch_job.sh --run_name "${NAME}_baseline"     	$ARGS
sbatch --output $OUT --job-name irm_001      		./scripts/sbatch_job.sh --run_name "${NAME}_irm_001"     	$ARGS   --irm.coef 0.01
sbatch --output $OUT --job-name irm_01       		./scripts/sbatch_job.sh --run_name "${NAME}_irm_01"      	$ARGS	--irm.coef 0.1
sbatch --output $OUT --job-name irm_1        		./scripts/sbatch_job.sh --run_name "${NAME}_irm_1"       	$ARGS	--irm.coef 1
sbatch --output $OUT --job-name mixup_001  		./scripts/sbatch_job.sh --run_name "${NAME}_mixup_001"   	$ARGS	--mixup.coef 0.01
sbatch --output $OUT --job-name mixup_01  		./scripts/sbatch_job.sh --run_name "${NAME}_mixup_01"   	$ARGS	--mixup.coef 0.1
sbatch --output $OUT --job-name mixup_1           	./scripts/sbatch_job.sh --run_name "${NAME}_mixup_1"   		$ARGS	--mixup.coef 1
sbatch --output $OUT --job-name rotation_001		./scripts/sbatch_job.sh --run_name "${NAME}_rotation_001"	$ARGS	--rotation.coef 0.01
sbatch --output $OUT --job-name rotation_001_nc     	./scripts/sbatch_job.sh --run_name "${NAME}_rotation_001_nc"	$ARGS	--rotation.coef 0.01 --rotation.compare False
sbatch --output $OUT --job-name rotation_01		./scripts/sbatch_job.sh --run_name "${NAME}_rotation_01"	$ARGS	--rotation.coef 0.1
sbatch --output $OUT --job-name rotation_01_nc		./scripts/sbatch_job.sh --run_name "${NAME}_rotation_01_nc"	$ARGS	--rotation.coef 0.1  --rotation.compare False
sbatch --output $OUT --job-name rotation_1		./scripts/sbatch_job.sh --run_name "${NAME}_rotation_1"		$ARGS	--rotation.coef 1
sbatch --output $OUT --job-name rotation_1_nc		./scripts/sbatch_job.sh --run_name "${NAME}_rotation_1_nc"	$ARGS	--rotation.coef 1    --rotation.compare False
sbatch --output $OUT --job-name brightness_001		./scripts/sbatch_job.sh --run_name "${NAME}_brightness_001"	$ARGS	--brightness.coef 0.01
sbatch --output $OUT --job-name brightness_001_nc	./scripts/sbatch_job.sh --run_name "${NAME}_brightness_001_nc"	$ARGS	--brightness.coef 0.01 --brightness.compare False
sbatch --output $OUT --job-name brightness_01		./scripts/sbatch_job.sh --run_name "${NAME}_brightness_01"	$ARGS	--brightness.coef 0.1
sbatch --output $OUT --job-name brightness_01_nc	./scripts/sbatch_job.sh --run_name "${NAME}_brightness_01_nc"	$ARGS	--brightness.coef 0.1  --brightness.compare False
sbatch --output $OUT --job-name brightness_1		./scripts/sbatch_job.sh --run_name "${NAME}_brightness_1"	$ARGS	--brightness.coef 1
sbatch --output $OUT --job-name brightness_1_nc      	./scripts/sbatch_job.sh --run_name "${NAME}_brightness_1_nc"	$ARGS	--brightness.coef 1    --brightness.compare False
sbatch --output $OUT --job-name simclr_001		./scripts/sbatch_job.sh --run_name "${NAME}_simclr_001"		$ARGS	--simclr.coef 0.01
sbatch --output $OUT --job-name simclr_01		./scripts/sbatch_job.sh --run_name "${NAME}_simclr_01"		$ARGS	--simclr.coef 0.1
sbatch --output $OUT --job-name simclr_1		./scripts/sbatch_job.sh	--run_name "${NAME}_simclr_1"		$ARGS	--simclr.coef 1
sbatch --output $OUT --job-name vae_001           	./scripts/sbatch_job.sh --run_name "${NAME}_vae_001"		$ARGS	--reconstruction.coef 0.01
sbatch --output $OUT --job-name vae_01			./scripts/sbatch_job.sh --run_name "${NAME}_vae_01"		$ARGS	--reconstruction.coef 0.1
sbatch --output $OUT --job-name vae_1			./scripts/sbatch_job.sh	--run_name "${NAME}_vae_1"		$ARGS	--reconstruction.coef 1

