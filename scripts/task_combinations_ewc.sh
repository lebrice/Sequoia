#!/bin/bash

./scripts/task_combinations.sh cifar100-10c_mh_ue100_se10       5 --dataset cifar100 --n_classes_per_task 10 --multihead --patience 5 --unsupervised_epochs_per_task 100 --supervised_epochs_per_task 10 --no_wandb_cleanup
./scripts/task_combinations.sh cifar100-10c_mh_ue100_se10_ewc   5 --dataset cifar100 --n_classes_per_task 10 --multihead --patience 5 --unsupervised_epochs_per_task 100 --supervised_epochs_per_task 10 --no_wandb_cleanup --ewc.coef 1
./scripts/task_combinations.sh cifar100-10c_mh_d_ue100_se10     5 --dataset cifar100 --n_classes_per_task 10 --multihead --patience 5 --unsupervised_epochs_per_task 100 --supervised_epochs_per_task 10 --no_wandb_cleanup --detach_classifier
./scripts/task_combinations.sh cifar100-10c_mh_d_ue100_se10_ewc 5 --dataset cifar100 --n_classes_per_task 10 --multihead --patience 5 --unsupervised_epochs_per_task 100 --supervised_epochs_per_task 10 --no_wandb_cleanup --detach_classifier --ewc.coef 1
