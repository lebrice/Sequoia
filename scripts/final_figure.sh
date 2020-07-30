#!/bin/bash
export SCRATCH=${SCRATCH:=$HOME}

COMMON_ARGS="\
    --unsupervised_epochs_per_task 0 \
    --supervised_epochs_per_task 200 \
    --multihead --no_wandb_cleanup \
    --n_neighbors 50 \
"
echo "Common args: $COMMON_ARGS"

source ./scripts/task_combinations_ewc.sh cifar100-20c 5 \
    $COMMON_ARGS \
    --dataset cifar100 --n_classes_per_task 20 \
    --tags cifar100-20c no-early-stopping

source ./scripts/task_combinations_ewc.sh cifar10 5 \
    $COMMON_ARGS \
    --dataset cifar10 \
    --tags cifar10 no-early-stopping replay

source ./scripts/task_combinations_ewc.sh fashion-mnist 5 \
    $COMMON_ARGS \
    --dataset fashion_mnist \
    --tags fashion-mnist no-early-stopping resnet-18

#source ./scripts/task_combinations_ewc.sh fashion-mnist 5 \
#    --unsupervised_epochs_per_task 0 --supervised_epochs_per_task 200 \
#    --use_accuracy_as_metric 1 \
#    --multihead --no_wandb_cleanup \
#    --dataset fashion_mnist \
#    --n_neighbors 50 \
#    --tags fashion-mnist
