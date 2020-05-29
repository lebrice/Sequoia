#!/bin/bash
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
NAME="${1:?'Name must be set'}"
N_JOBS="${2:?'N_JOBS must be set'}"
OUT="$SCRATCH/slurm_out/$NAME/%x-%j.out"
ARGS=${@:3}

mkdir -p "$SCRATCH/slurm_out/$NAME"
echo "Launching sbatch --output $OUT --job-name $NAME --array=1-$N_JOBS run_ssl_cc.sh $ARGS"
sbatch --output $OUT --job-name $NAME --array=1-$N_JOBS run_ssl_cc.sh $ARGS

