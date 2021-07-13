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
PROJECT="crl_study"

SETTINGS=(
    "continual_rl"
    "discrete_task_agnostic_rl"
    "incremental_rl"
    "task_incremental_rl"
    "multi_task_rl"
    "traditional_rl"
)
METHODS=(
    "ppo"
    "a2c"
    "dqn"
    "ddpg"
    "sac"
    "td3"
    "baseline"
    "methods.ewc"
)
BENCHMARKS=(
    "cartpole"
    "monsterkong_mix"
    "mountaincar_continuous"
)
# "half_cheetah"

for METHOD in "${METHODS[@]}"; do
    for SETTING in "${SETTINGS[@]}"; do
        for BENCHMARK in "${BENCHMARKS[@]}"; do
            # Share the trials from different datasets, hopefully reusing something?
            DATABASE_PATH="/mnt/home/${SETTING}_${METHOD}.pkl"
            scripts/eai/job.sh sequoia_sweep \
                --max_runs $MAX_RUNS --database_path $DATABASE_PATH \
                --setting $SETTING --benchmark $BENCHMARK --project $PROJECT \
                --method $METHOD \
                "$@"
        done
    done
done

# source scripts/eai/job.sh sequoia_sweep --max_runs 20 --database_path /mnt/home/orion_db.pkl --setting class_incremental --dataset cifar10  --project csl_study --method baseline
# source scripts/eai/job.sh sequoia_sweep --max_runs 20 --database_path /mnt/home/orion_db.pkl --setting class_incremental --dataset cifar100 --project csl_study --nb_tasks 20 --method baseline
# source scripts/eai/job.sh sequoia_sweep --max_runs 20 --database_path /mnt/home/orion_db.pkl --setting class_incremental --dataset synbols  --project csl_study --nb_tasks 12 --method baseline
