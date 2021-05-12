#!/bin/bash
set -o errexit  # Used to exit upon error, avoiding cascading errors
set -o errtrace # Show error trace
set -o pipefail # Unveils hidden failures
set -o nounset  # Exposes unset variables
export WANDB_API_KEY=${WANDB_API_KEY?"Need to pass the wandb api key or have it set in the environment variables."}

module load anaconda/3
conda activate sequoia

cd ~/Sequoia
pip install -e .[hpo,monsterkong]

# Number of runs per combination.
MAX_RUNS=20
PROJECT="csl_study"

SETTINGS=("class_incremental" "task_incremental" "multi_task" "iid")
METHODS=(
    "gdumb" "random_baseline" "pnn" "agem"
    "ar1" "cwr_star" "gem" "gdumb" "lwf" "replay" "synaptic_intelligence"
    "avalanche.ewc" "methods.ewc" "experience_replay" "hat" "baseline"
)
DATASETS=(
    "synbols --nb_tasks 12"
    "cifar10"
    "cifar100 --nb_tasks 10"
    "mnist"
)

for METHOD in "${METHODS[@]}"; do
    for SETTING in "${SETTINGS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            # Share the trials from different datasets, hopefully reusing something?
            DABASE_PATH="/mnt/home/${SETTING}_${METHOD}.pkl"
            scripts/slurm/sweep.sh \
                --max_runs $MAX_RUNS --database_path $DABASE_PATH \
                --setting $SETTING --dataset $DATASET --project $PROJECT \
                --WANDB_API_KEY $WANDB_API_KEY \
                --method $METHOD \
                "$@"
        done
    done
done
