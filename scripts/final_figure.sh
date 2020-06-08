#!/bin/bash

export SCRATCH=${SCRATCH:=$HOME}


source ./scripts/task_combinations_ewc.sh cifar100-20c 5 \
    --unsupervised_epochs_per_task 0 --supervised_epochs_per_task 200 \
    --multihead --no_wandb_cleanup \
    --dataset cifar100 --n_classes_per_task 20 \
    --n_neighbors 50 \
    --tags cifar100-20c no-early-stopping

source ./scripts/task_combinations_ewc.sh cifar10 5 \
    --unsupervised_epochs_per_task 0 --supervised_epochs_per_task 200 \
    --multihead --no_wandb_cleanup \
    --dataset cifar10 \
    --n_neighbors 50 \
    --tags cifar10 no-early-stopping

source ./scripts/task_combinations_ewc.sh fashion-mnist 5 \
    --unsupervised_epochs_per_task 0 --supervised_epochs_per_task 200 \
    --multihead --no_wandb_cleanup \
    --dataset fashion_mnist \
    --n_neighbors 50 \
    --tags fashion-mnist no-early-stopping

