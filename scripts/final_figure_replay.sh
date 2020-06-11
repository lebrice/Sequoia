#!/bin/bash

export SCRATCH=${SCRATCH:=$HOME}

COMMON_ARGS="\
    --unsupervised_epochs_per_task 0 \
    --supervised_epochs_per_task 200 \
    --replay_buffer_size 2000 \
    --no_wandb_cleanup \
    --n_neighbors 50 \
    --tags no-early-stopping replay \
"
echo "Common args: $COMMON_ARGS"

source ./scripts/task_combinations_ewc.sh cifar100-20c 5 \
    $COMMON_ARGS \
    --dataset cifar100 --n_classes_per_task 20

source ./scripts/task_combinations_ewc.sh cifar10 5 \
    $COMMON_ARGS \
    --dataset cifar10

source ./scripts/task_combinations_ewc.sh fashion-mnist 5 \
    $COMMON_ARGS \
    --dataset fashion_mnist

