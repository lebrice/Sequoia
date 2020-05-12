#!/bin/bash
module load httpproxy
module load python/3.7

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
NAME="${1:?'Name must be set'}"
N_JOBS="${2:?'N_JOBS must be set'}"
OUT="$SCRATCH/slurm_out/$NAME/%x-%j.out"
ARGS=${@:3}

echo "sbatch with name '$NAME' and with args '$ARGS'"
echo "Executing main.py with additional args: ${@:1}"
# Create the slurm output dir if it doesn't exist already.

# activate the virtual environment (only used to download the datasets)
source ~/ENV/bin/activate
python -m scripts.download_datasets --data_dir "$SCRATCH/data"
python -m scripts.download_pretrained_models # --save_dir "$SCRATCH/checkpoints"
deactivate

# Zip up the dataset, if not already present.
zip -u "$SCRATCH/data.zip" "$SCRATCH/data"

mkdir -p "$SCRATCH/slurm_out/$NAME"

echo "Launching sbatch_job.sh with additional args: '--output $OUT --job-name $NAME --array=1-$N_JOBS ./scripts/sbatch_job.sh $ARGS'"

sbatch --output $OUT --job-name $NAME --array=1-$N_JOBS ./scripts/sbatch_job.sh $ARGS
