#!/bin/bash
set -o errexit  # Used to exit upon error, avoiding cascading errors
set -o errtrace # Show error trace
set -o pipefail # Unveils hidden failures
set -o nounset  # Exposes unset variables
export WANDB_API_KEY=${WANDB_API_KEY?"Need to pass the wandb api key or have it set in the environment variables."}

source dockers/eai/build.sh

export NO_BUILD=1

# Number of runs per combination.
MAX_RUNS=20
PROJECT="csl_study"

SETTINGS=(
    "continual_sl"
    "discrete_task_agnostic_sl"
    "incremental_sl"
    # "task_incremental_sl"
    # "multi_task_sl"
    # "traditional_sl"
)
METHODS=(
    # "random_baseline"
    "naive"
    "gdumb"
    "agem"
    "ar1"
    "cwr_star"
    "gem"
    "lwf"
    "replay"
    "synaptic_intelligence"
    "avalanche.ewc"
    "baseline"
    "methods.ewc"
    "experience_replay"
    "hat"
    "pnn"
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
            scripts/eai/job.sh sequoia_sweep \
                --max_runs $MAX_RUNS --database_path $DABASE_PATH \
                --setting $SETTING --dataset $DATASET --project $PROJECT \
                --method $METHOD --monitor_training_performance True \
                "$@"
        done
    done
done

# source scripts/eai/job.sh sequoia_sweep --max_runs 20 --database_path /mnt/home/orion_db.pkl --setting class_incremental --dataset cifar10  --project csl_study --method baseline
# source scripts/eai/job.sh sequoia_sweep --max_runs 20 --database_path /mnt/home/orion_db.pkl --setting class_incremental --dataset cifar100 --project csl_study --nb_tasks 20 --method baseline
# source scripts/eai/job.sh sequoia_sweep --max_runs 20 --database_path /mnt/home/orion_db.pkl --setting class_incremental --dataset synbols  --project csl_study --nb_tasks 12 --method baseline
