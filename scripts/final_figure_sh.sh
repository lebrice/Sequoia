#!/bin/bash

export SCRATCH=${SCRATCH:=$HOME}


source ./scripts/task_combinations_ewc.sh cifar100-20c-sh 5 \
    --unsupervised_epochs_per_task --supervised_epochs_per_task 200 \
    --no_wandb_cleanup \
    --dataset cifar100 --n_classes_per_task 20 \
    --n_neighbors 50 \
    --tags cifar100-20c single-head

source ./scripts/task_combinations_ewc.sh cifar10-sh 5 \
    --unsupervised_epochs_per_task 0 --supervised_epochs_per_task 200 \
    --no_wandb_cleanup \
    --dataset cifar10 \
    --n_neighbors 50 \
    --tags cifar10 single-head

source ./scripts/task_combinations_ewc.sh fashion-mnist-sh 5 \
    --unsupervised_epochs_per_task 0 --supervised_epochs_per_task 200 \
    --no_wandb_cleanup \
    --dataset fashion_mnist \
    --n_neighbors 50 \
    --tags fashion-mnist single-head
