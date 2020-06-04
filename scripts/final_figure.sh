!/bin/bash

source ./scripts/task_combinations_ewc.sh \
    --unsupervised_epochs_per_task 0 --supervised_epochs_per_task 100 \
    --patience 10 \
    --multihead --no_wandb_cleanup \
    --dataset cifar100 --n_classes_per_task 20 \
    --run_group cifar100-20c \
    --n_neighbours 50 \
    --tags cifar100-20c

