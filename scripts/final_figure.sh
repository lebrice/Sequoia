#!/bin/bash
export SCRATCH=${SCRATCH:=$HOME}

cp -r /home/ostapeno/dev/SSCL/* /home/ostapeno/projects/rrg-bengioy-ad/ostapeno/dev/SSCL/
cd /home/ostapeno/projects/rrg-bengioy-ad/ostapeno/dev/SSCL


source ./scripts/task_combinations_ewc.sh cifar100-20c 3 \
    --unsupervised_epochs_per_task 0 --supervised_epochs_per_task 200 \
    --multihead --no_wandb_cleanup \
    --dataset cifar100 --n_classes_per_task 20 \
    --patience 10 \
    --n_neighbors 50 \
    --random_class_ordering 0 \
    --project_name SSCL_ewc \
    --tags cifar100-20c

#source ./scripts/task_combinations_ewc.sh cifar10 5 \
#    --unsupervised_epochs_per_task 0 --supervised_epochs_per_task 200 \
#    --multihead --no_wandb_cleanup \
#    --dataset cifar10 \
#    --n_neighbors 50 \
#    --tags cifar10 no-early-stopping

#source ./scripts/task_combinations_ewc.sh fashion-mnist 5 \
#    --unsupervised_epochs_per_task 0 --supervised_epochs_per_task 200 \
#    --use_accuracy_as_metric 1 \
#    --multihead --no_wandb_cleanup \
#    --dataset fashion_mnist \
#    --n_neighbors 50 \
#    --tags fashion-mnist
