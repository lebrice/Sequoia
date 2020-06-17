#!/bin/bash
# Script that launches the right sbatch command depending on which cluster we are on.
# the job name.
NAME="${1:?'Name must be set'}"
N_JOBS="${2:?'N_JOBS must be set'}"
# Set $SCRATCH to $HOME if it isn't set. (only has an effect when running on mila cluster)
export SCRATCH=${SCRATCH:=$HOME}
OUT="$SCRATCH/slurm_out/$NAME/%x-%A_%a.out"

# Capture the other arguments to the script, they will be passed to the launched sbatch script.
ARGS="${@:3}"

echo "OUT pattern: $OUT"
echo "Number of jobs per task: $N_JOBS"

# Create the slurm output dir if it doesn't exist already.
mkdir -p "$SCRATCH/slurm_out/$NAME"

if [[ $HOSTNAME == *"beluga"* ]]; then
    echo "Launching \
    sbatch --output $OUT --job-name $NAME --time 24:00:00 --array=1-$N_JOBS ./scripts/beluga/run.sh $ARGS"
    sbatch --output $OUT --job-name $NAME --time 24:00:00 --array=1-$N_JOBS ./scripts/beluga/run.sh $ARGS
elif [[ $HOSTNAME == *"cedar"* ]]; then
    echo "Launching \
    sbatch --output $OUT --job-name $NAME --time 24:00:00 --array=1-$N_JOBS ./scripts/cedar/run.sh $ARGS"
    sbatch --output $OUT --job-name $NAME --time 24:00:00 --array=1-$N_JOBS ./scripts/cedar/run.sh $ARGS
else
    echo "Launching \
    sbatch --output $OUT --job-name $NAME --time 48:00:00 --array=1-$N_JOBS ./scripts/cedar/run.sh $ARGS"
    sbatch --output $OUT --job-name $NAME --time 48:00:00 --array=1-$N_JOBS ./scripts/cedar/run.sh $ARGS
fi